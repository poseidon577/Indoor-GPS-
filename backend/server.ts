import express from "express";
import path from "path";
import fs from "fs";
import { exec, spawn } from "child_process";
import { createServer as createViteServer } from "vite";

type TelemetryValue = string | number | boolean | null;
type TelemetryRow = Record<string, TelemetryValue>;
type TelemetrySheet = "overview" | "environment" | "optical_channel" | "communication" | "localization" | "subcarriers" | "power" | "optimization" | "validation" | "events" | "run_metadata";
type ResultRun = { runId: string; latestFrame: number; lastSimulationTime: number; createdAt: string; sheets: Map<TelemetrySheet, TelemetryRow[]>; metadata: TelemetryRow; listeners: Set<any> };
const RESULT_SHEETS: TelemetrySheet[] = ["overview", "environment", "optical_channel", "communication", "localization", "subcarriers", "power", "optimization", "validation", "events", "run_metadata"];
const resultRuns = new Map<string, ResultRun>();

function makeRun(runId: string): ResultRun {
  const createdAt = new Date().toISOString();
  return { runId, latestFrame: -1, lastSimulationTime: -1, createdAt, sheets: new Map(RESULT_SHEETS.map((name) => [name, []])), metadata: { run_id: runId, telemetry_schema_version: "1.0.0", start_timestamp: createdAt }, listeners: new Set() };
}

function safeTelemetryValue(value: unknown): TelemetryValue {
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function sanitizeRows(value: unknown): TelemetryRow[] {
  if (!Array.isArray(value)) return [];
  return value.filter((row): row is Record<string, unknown> => !!row && typeof row === "object" && !Array.isArray(row)).map((row) => Object.fromEntries(Object.entries(row).map(([key, item]) => [key, safeTelemetryValue(item)])));
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  const BASE_DIR = fs.existsSync(path.join(__dirname, "package.json"))
    ? __dirname
    : path.join(__dirname, "..");

  app.use(express.json());

  // Live Results telemetry store. It is an observer: rows arrive from existing
  // module states and are committed atomically without invoking any simulator code.
  app.get("/api/results/runs", (_req, res) => {
    res.json([...resultRuns.values()].map((run) => ({ run_id: run.runId, latest_frame: run.latestFrame, created_at: run.createdAt, rows: Object.fromEntries([...run.sheets].map(([name, rows]) => [name, rows.length])) })));
  });

  app.post("/api/results/:runId/frame", (req, res) => {
    const runId = String(req.params.runId);
    const frame = req.body;
    if (!frame || !Number.isInteger(frame.frame_id) || frame.frame_id < 0 || typeof frame.simulation_time_s !== "number" || !Number.isFinite(frame.simulation_time_s) || !frame.sheets || typeof frame.sheets !== "object") {
      return res.status(400).json({ error: "Invalid telemetry frame envelope." });
    }
    const run = resultRuns.get(runId) ?? makeRun(runId);
    if (!resultRuns.has(runId)) resultRuns.set(runId, run);
    if (frame.frame_id <= run.latestFrame) return res.status(409).json({ error: "Frame ID must be unique and strictly increasing." });
    if (frame.simulation_time_s < run.lastSimulationTime) return res.status(400).json({ error: "Simulation time must be monotonic." });
    const prepared = new Map<TelemetrySheet, TelemetryRow[]>();
    for (const sheet of RESULT_SHEETS) {
      if (sheet === "run_metadata") continue;
      const rows = sanitizeRows(frame.sheets[sheet]);
      if (rows.some((row) => row.run_id !== runId || row.frame_id !== frame.frame_id)) return res.status(400).json({ error: `Rows in ${sheet} must share the envelope run_id and frame_id.` });
      prepared.set(sheet, rows);
    }
    // Commit every prepared sheet only after validation succeeds: no partial frames.
    for (const [sheet, rows] of prepared) run.sheets.get(sheet)!.push(...rows);
    run.latestFrame = frame.frame_id;
    run.lastSimulationTime = frame.simulation_time_s;
    run.metadata = { ...run.metadata, last_committed_frame: frame.frame_id, last_simulation_time_s: frame.simulation_time_s, last_wall_clock_timestamp: typeof frame.wall_clock_timestamp === "string" ? frame.wall_clock_timestamp : null };
    const event = `event: frame\ndata: ${JSON.stringify({ run_id: runId, frame_id: frame.frame_id, simulation_time_s: frame.simulation_time_s, row_counts: Object.fromEntries([...prepared].map(([name, rows]) => [name, rows.length])) })}\n\n`;
    for (const listener of run.listeners) listener.write(event);
    res.status(201).json({ committed: true, run_id: runId, frame_id: frame.frame_id });
  });

  app.get("/api/results/:runId/summary", (req, res) => {
    const run = resultRuns.get(req.params.runId);
    if (!run) return res.status(404).json({ error: "Run not found." });
    res.json({ run_id: run.runId, latest_frame: run.latestFrame, row_counts: Object.fromEntries([...run.sheets].map(([name, rows]) => [name, rows.length])), metadata: run.metadata });
  });

  app.get("/api/results/:runId/sheet/:sheet", (req, res) => {
    const run = resultRuns.get(req.params.runId), sheet = req.params.sheet as TelemetrySheet;
    if (!run) return res.status(404).json({ error: "Run not found." });
    if (!RESULT_SHEETS.includes(sheet)) return res.status(404).json({ error: "Unknown worksheet." });
    const limit = Math.min(Math.max(Number(req.query.limit ?? 200), 1), 5000), offset = Math.max(Number(req.query.offset ?? 0), 0);
    const filter = String(req.query.search ?? "").toLowerCase();
    const rows = run.sheets.get(sheet)!.filter((row) => !filter || Object.values(row).some((value) => String(value ?? "").toLowerCase().includes(filter)));
    res.json({ run_id: run.runId, sheet, total: rows.length, offset, limit, rows: rows.slice(offset, offset + limit) });
  });

  app.get("/api/results/:runId/frame/:frameId", (req, res) => {
    const run = resultRuns.get(req.params.runId), frameId = Number(req.params.frameId);
    if (!run) return res.status(404).json({ error: "Run not found." });
    res.json({ run_id: run.runId, frame_id: frameId, sheets: Object.fromEntries(RESULT_SHEETS.map((sheet) => [sheet, run.sheets.get(sheet)!.filter((row) => row.frame_id === frameId)])) });
  });

  app.get("/api/results/:runId/export", (req, res) => {
    const run = resultRuns.get(req.params.runId);
    if (!run) return res.status(404).json({ error: "Run not found." });
    // A JSON copy is an immutable export snapshot at the last committed frame.
    const snapshot = { run_id: run.runId, frame_id: run.latestFrame, metadata: { ...run.metadata, export_timestamp: new Date().toISOString() }, rows: Object.fromEntries(RESULT_SHEETS.map((sheet) => [sheet, [...run.sheets.get(sheet)!]])) };
    res.json(snapshot);
  });

  app.get("/api/results/:runId/stream", (req, res) => {
    const run = resultRuns.get(req.params.runId) ?? makeRun(req.params.runId);
    if (!resultRuns.has(req.params.runId)) resultRuns.set(req.params.runId, run);
    res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" });
    res.write(`event: ready\ndata: ${JSON.stringify({ run_id: run.runId, latest_frame: run.latestFrame })}\n\n`);
    run.listeners.add(res);
    req.on("close", () => run.listeners.delete(res));
  });

  // API Route: Get configuration
  app.get("/api/config", (req, res) => {
    const configPath = path.join(BASE_DIR, "VLCL_AI", "configs", "default.yaml");
    if (fs.existsSync(configPath)) {
      res.send(fs.readFileSync(configPath, "utf-8"));
    } else {
      // Fallback with default content if it doesn't exist yet
      const defaultConfig = `room:
  width: 5.0
  length: 5.0
  height: 3.0
  wall_reflectivity: 0.8
  floor_reflectivity: 0.2
  ceiling_reflectivity: 0.5

leds:
  - id: 1
    position: [1.25, 1.25, 3.0]
    orientation: [0, 0, -1]
    power: 20.0
    bias_current: 0.5
    frequency: 100000.0
    lambertian_order: 1.0
    beam_angle: 60.0
    fov: 60.0
  - id: 2
    position: [3.75, 1.25, 3.0]
    orientation: [0, 0, -1]
    power: 20.0
    bias_current: 0.5
    frequency: 150000.0
    lambertian_order: 1.0
    beam_angle: 60.0
    fov: 60.0
  - id: 3
    position: [1.25, 3.75, 3.0]
    orientation: [0, 0, -1]
    power: 20.0
    bias_current: 0.5
    frequency: 200000.0
    lambertian_order: 1.0
    beam_angle: 60.0
    fov: 60.0
  - id: 4
    position: [3.75, 3.75, 3.0]
    orientation: [0, 0, -1]
    power: 20.0
    bias_current: 0.5
    frequency: 250000.0
    lambertian_order: 1.0
    beam_angle: 60.0
    fov: 60.0

receiver:
  position: [2.5, 2.5, 0.85]
  orientation: [0, 0, 1]
  velocity: [0.2, 0.1, 0.0]
  acceleration: [0.0, 0.0, 0.0]
  fov: 70.0
  apd_size: 1e-4
  noise: 1e-14
  gain: 1.0

mobility:
  type: "circular" # static, linear, circular, random_walk, waypoint, spline
  speed: 0.5
  radius: 1.5
  center: [2.5, 2.5, 0.85]

obstacles:
  - id: "obs_human"
    type: "cylinder" # box, cylinder, sphere
    position: [2.0, 3.0, 0.9]
    rotation: [0, 0, 0]
    scale: [0.3, 0.3, 1.8] # x_radius, y_radius, height
    reflectivity: 0.3
    material: "skin_fabric"
`;
      res.send(defaultConfig);
    }
  });

  // API Route: Save configuration
  app.post("/api/config", (req, res) => {
    const { content } = req.body;
    const configPath = path.join(BASE_DIR, "VLCL_AI", "configs", "default.yaml");
    try {
      fs.mkdirSync(path.dirname(configPath), { recursive: true });
      fs.writeFileSync(configPath, content, "utf-8");
      res.json({ success: true });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  // API Route: Run python simulation
  app.post("/api/run", (req, res) => {
    const scriptPath = path.join(BASE_DIR, "VLCL_AI", "examples", "demo_environment.py");

    // Ensure directories exist
    fs.mkdirSync(path.join(BASE_DIR, "VLCL_AI", "logs"), { recursive: true });

    // Execute python script
    const isWin = process.platform === "win32";
    const venvPythonPath = isWin
      ? path.join(BASE_DIR, "VLCL_AI", ".venv", "Scripts", "python.exe")
      : path.join(BASE_DIR, "VLCL_AI", ".venv", "bin", "python3");
    const pythonCmd = fs.existsSync(venvPythonPath) ? venvPythonPath : (isWin ? "python" : "python3");

    exec(`${pythonCmd} "${scriptPath}"`, { cwd: BASE_DIR, env: { ...process.env, PYTHONIOENCODING: "utf-8" } }, (error, stdout, stderr) => {
      res.json({
        success: !error,
        stdout,
        stderr,
        error: error ? error.message : null
      });
    });
  });

  // API Route: Real-time Physics Engine computation
  app.post("/api/physics", (req, res) => {
    const envState = req.body;

    const isWin = process.platform === "win32";
    const venvPythonPath = isWin
      ? path.join(BASE_DIR, "VLCL_AI", ".venv", "Scripts", "python.exe")
      : path.join(BASE_DIR, "VLCL_AI", ".venv", "bin", "python3");
    const pythonCmd = fs.existsSync(venvPythonPath) ? venvPythonPath : (isWin ? "python" : "python3");

    // Inline Python script that instantiates PhysicsEngine and computes metrics
    const inlineScript = `
import sys, os, json, math
sys.path.insert(0, r"${BASE_DIR.replace(/\\/g, "\\\\")}")
from VLCL_AI.environment.state import EnvironmentState
from VLCL_AI.physics.physics_engine import PhysicsEngine

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

state = EnvironmentState(
    current_time=data.get("current_time", 0.0),
    frame_index=data.get("frame_index", 0),
    fps=data.get("fps", 60.0),
    receiver_position=data["receiver_position"],
    receiver_orientation=data["receiver_orientation"],
    receiver_velocity=data.get("receiver_velocity", [0,0,0]),
    receiver_acceleration=data.get("receiver_acceleration", [0,0,0]),
    receiver_angles=data.get("receiver_angles", {"roll":0,"pitch":0,"yaw":0}),
    room_dims=data.get("room_dims", [5.0, 5.0, 3.0]),
    led_positions={int(k): v for k,v in data["led_positions"].items()},
    led_powers={int(k): v for k,v in data["led_powers"].items()},
    led_active={int(k): v for k,v in data["led_active"].items()},
    led_orientations={int(k): v for k,v in data["led_orientations"].items()},
    led_beam_angles={int(k): v for k,v in data["led_beam_angles"].items()},
    distances={int(k): v for k,v in data["distances"].items()},
    incident_angles_rad={int(k): v * math.pi / 180.0 for k,v in data["incident_angles"].items()},
    irradiance_angles_rad={int(k): v * math.pi / 180.0 for k,v in data["irradiance_angles"].items()},
    visibility_matrix={int(k): v for k,v in data["visibility_matrix"].items()},
    los_matrix={int(k): v for k,v in data["los_matrix"].items()},
    blocking_obstacles={int(k): v for k,v in data["blocking_obstacles"].items()},
    obstacles=data.get("obstacles", [])
)

engine = PhysicsEngine()
engine.compute(state)
result = engine.export()

def str_keys(d):
    return {str(k): v for k,v in d.items()} if isinstance(d, dict) else d

output = {
    "snrs": str_keys(result.get("snrs", {})),
    "received_powers": str_keys(result.get("received_powers", {})),
    "los_gains": str_keys(result.get("los_gains", {})),
    "nlos_gains": str_keys(result.get("nlos_gains", {})),
    "electrical_currents": str_keys(result.get("electrical_currents", {})),
    "voltages": str_keys(result.get("voltages", {})),
    "metrics": result.get("metrics", {})
}
print(json.dumps(output))
`;

    // Use a temp file approach for reliability (avoids shell escaping issues)

    const tmpScript = path.join(BASE_DIR, "VLCL_AI", "logs", "_physics_tmp.py");
    const tmpInput = path.join(BASE_DIR, "VLCL_AI", "logs", "_physics_input.json");

    try {
      fs.mkdirSync(path.join(BASE_DIR, "VLCL_AI", "logs"), { recursive: true });
      fs.writeFileSync(tmpScript, inlineScript, "utf-8");
      fs.writeFileSync(tmpInput, JSON.stringify(envState), "utf-8");
    } catch (e: any) {
      return res.status(500).json({ error: "Failed to write temp files: " + e.message });
    }

    exec(
      `${pythonCmd} "${tmpScript}" "${tmpInput}"`,
      { cwd: BASE_DIR, env: { ...process.env, PYTHONIOENCODING: "utf-8", LOGURU_LEVEL: "WARNING" }, timeout: 8000 },
      (error, stdout, stderr) => {
        if (error) {
          return res.status(500).json({ error: stderr || error.message });
        }
        try {
          const result = JSON.parse(stdout.trim());
          res.json({ success: true, physics: result });
        } catch {
          res.status(500).json({ error: "Failed to parse physics output", raw: stdout });
        }
      }
    );
  });

  // API Route: Real-time Communication Engine computation (Module 3)
  // Chains Physics → Communication Engine and returns OFDM/VLC KPIs
  app.post("/api/communication", (req, res) => {
    const envState = req.body;

    const isWin = process.platform === "win32";
    const venvPythonPath = isWin
      ? path.join(BASE_DIR, "VLCL_AI", ".venv", "Scripts", "python.exe")
      : path.join(BASE_DIR, "VLCL_AI", ".venv", "bin", "python3");
    const pythonCmd = fs.existsSync(venvPythonPath) ? venvPythonPath : (isWin ? "python" : "python3");

    const inlineScript = `
import sys, os, json, math
sys.path.insert(0, r"${BASE_DIR.replace(/\\/g, "\\\\")}")
from VLCL_AI.environment.state import EnvironmentState
from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.communication.engine import CommunicationEngine

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

env_state = EnvironmentState(
    current_time=data.get("current_time", 0.0),
    frame_index=data.get("frame_index", 0),
    fps=data.get("fps", 60.0),
    receiver_position=data["receiver_position"],
    receiver_orientation=data["receiver_orientation"],
    receiver_velocity=data.get("receiver_velocity", [0,0,0]),
    receiver_acceleration=data.get("receiver_acceleration", [0,0,0]),
    receiver_angles=data.get("receiver_angles", {"roll":0,"pitch":0,"yaw":0}),
    room_dims=data.get("room_dims", [5.0, 5.0, 3.0]),
    led_positions={int(k): v for k,v in data["led_positions"].items()},
    led_powers={int(k): v for k,v in data["led_powers"].items()},
    led_active={int(k): v for k,v in data["led_active"].items()},
    led_orientations={int(k): v for k,v in data["led_orientations"].items()},
    led_beam_angles={int(k): v for k,v in data["led_beam_angles"].items()},
    distances={int(k): v for k,v in data["distances"].items()},
    incident_angles_rad={int(k): v * math.pi / 180.0 for k,v in data["incident_angles"].items()},
    irradiance_angles_rad={int(k): v * math.pi / 180.0 for k,v in data["irradiance_angles"].items()},
    visibility_matrix={int(k): v for k,v in data["visibility_matrix"].items()},
    los_matrix={int(k): v for k,v in data["los_matrix"].items()},
    blocking_obstacles={int(k): v for k,v in data["blocking_obstacles"].items()},
    obstacles=data.get("obstacles", [])
)

# Step 1: Run Physics Engine to get PhysicsState
physics_engine = PhysicsEngine()
physics_state = physics_engine.compute(env_state)

# Step 2: Run Communication Engine using Physics results
comm_engine = CommunicationEngine()
comm_state = comm_engine.step(env_state, physics_state)

# Return lightweight summary dict safe for JSON
print(json.dumps(comm_state.to_summary_dict()))
`;

    const tmpScript = path.join(BASE_DIR, "VLCL_AI", "logs", "_comm_tmp.py");
    const tmpInput = path.join(BASE_DIR, "VLCL_AI", "logs", "_comm_input.json");

    try {
      fs.mkdirSync(path.join(BASE_DIR, "VLCL_AI", "logs"), { recursive: true });
      fs.writeFileSync(tmpScript, inlineScript, "utf-8");
      fs.writeFileSync(tmpInput, JSON.stringify(envState), "utf-8");
    } catch (e: any) {
      return res.status(500).json({ error: "Failed to write temp files: " + e.message });
    }

    exec(
      `${pythonCmd} "${tmpScript}" "${tmpInput}"`,
      { cwd: BASE_DIR, env: { ...process.env, PYTHONIOENCODING: "utf-8", LOGURU_LEVEL: "WARNING" }, timeout: 15000 },
      (error, stdout, stderr) => {
        if (error) {
          return res.status(500).json({ error: stderr || error.message });
        }
        try {
          const outStr = stdout.trim();
          const jsonStr = outStr.substring(outStr.indexOf('{'));
          const result = JSON.parse(jsonStr);
          res.json({ success: true, communication: result });
        } catch {
          res.status(500).json({ error: "Failed to parse communication output", raw: stdout });
        }
      }
    );
  });


  // API Route: Real-time Localization Engine computation (Module 4 — A-DPDOA)
  // Chains Physics → LocalizationEngine and returns A-DPDOA position estimate + metrics
  app.post("/api/localization", (req, res) => {
    const envState = req.body;

    const isWin = process.platform === "win32";
    const venvPythonPath = isWin
      ? path.join(BASE_DIR, "VLCL_AI", ".venv", "Scripts", "python.exe")
      : path.join(BASE_DIR, "VLCL_AI", ".venv", "bin", "python3");
    const pythonCmd = fs.existsSync(venvPythonPath) ? venvPythonPath : (isWin ? "python" : "python3");

    const inlineScript = `
import sys, os, json, math
try:
    sys.path.insert(0, r"${BASE_DIR.replace(/\\/g, "\\\\")}")
    from VLCL_AI.environment.state import EnvironmentState
    from VLCL_AI.physics.physics_engine import PhysicsEngine
    from VLCL_AI.localization.engine import LocalizationEngine

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    env_state = EnvironmentState(
        current_time=data.get("current_time", 0.0),
        frame_index=data.get("frame_index", 0),
        fps=data.get("fps", 60.0),
        receiver_position=data["receiver_position"],
        receiver_orientation=data["receiver_orientation"],
        receiver_velocity=data.get("receiver_velocity", [0,0,0]),
        receiver_acceleration=data.get("receiver_acceleration", [0,0,0]),
        receiver_angles=data.get("receiver_angles", {"roll":0,"pitch":0,"yaw":0}),
        room_dims=data.get("room_dims", [5.0, 5.0, 3.0]),
        led_positions={int(k): v for k,v in data["led_positions"].items()},
        led_powers={int(k): v for k,v in data["led_powers"].items()},
        led_active={int(k): v for k,v in data["led_active"].items()},
        led_orientations={int(k): v for k,v in data["led_orientations"].items()},
        led_beam_angles={int(k): v for k,v in data["led_beam_angles"].items()},
        distances={int(k): v for k,v in data["distances"].items()},
        incident_angles_rad={int(k): v * math.pi / 180.0 for k,v in data["incident_angles"].items()},
        irradiance_angles_rad={int(k): v * math.pi / 180.0 for k,v in data["irradiance_angles"].items()},
        visibility_matrix={int(k): v for k,v in data["visibility_matrix"].items()},
        los_matrix={int(k): v for k,v in data["los_matrix"].items()},
        blocking_obstacles={int(k): v for k,v in data["blocking_obstacles"].items()},
        obstacles=data.get("obstacles", [])
    )

    # Step 1: Run Physics Engine to get PhysicsState
    physics_engine = PhysicsEngine()
    physics_state = physics_engine.compute(env_state)

    # Step 2: Run Localization Engine
    loc_engine = LocalizationEngine()
    loc_state = loc_engine.step(env_state, physics_state)

    # Convert dict keys to str for JSON serialisation
    result = loc_state.to_dict()
    def str_keys(d):
        return {str(k): v for k, v in d.items()} if isinstance(d, dict) else d

    result["signals"]["received_powers"] = str_keys(result["signals"]["received_powers"])
    result["signals"]["localization_snr"] = str_keys(result["signals"]["localization_snr"])

    print(json.dumps(result))
except Exception as e:
    import traceback
    print(json.dumps({"__error__": str(e), "__traceback__": traceback.format_exc()}))
    sys.exit(1)
`;

    const tmpScript = path.join(BASE_DIR, "VLCL_AI", "logs", "_loc_tmp.py");
    const tmpInput = path.join(BASE_DIR, "VLCL_AI", "logs", "_loc_input.json");

    try {
      fs.mkdirSync(path.join(BASE_DIR, "VLCL_AI", "logs"), { recursive: true });
      fs.writeFileSync(tmpScript, inlineScript, "utf-8");
      fs.writeFileSync(tmpInput, JSON.stringify(envState), "utf-8");
    } catch (e: any) {
      return res.status(500).json({ error: "Failed to write temp files: " + e.message });
    }

    exec(
      `${pythonCmd} "${tmpScript}" "${tmpInput}"`,
      { cwd: BASE_DIR, env: { ...process.env, PYTHONIOENCODING: "utf-8", LOGURU_LEVEL: "WARNING" }, timeout: 30000 },
      (error, stdout, stderr) => {
        if (error && !stdout.trim()) {
          return res.status(500).json({ error: stderr || error.message });
        }
        try {
          const result = JSON.parse(stdout.trim());
          // Python emitted a caught exception — treat as error
          if (result.__error__) {
            return res.status(500).json({ error: result.__error__, traceback: result.__traceback__ });
          }
          res.json({ success: true, localization: result });
        } catch {
          res.status(500).json({ error: "Failed to parse localization output", raw: stdout.slice(0, 500) });
        }
      }
    );
  });


  // API Route: Real-time Integrated VLCL Engine computation (Module 5)
  // Chains Physics -> IntegratedVLCLEngine and returns composite communication + localization metrics
  app.post("/api/integrated", (req, res) => {
    const envState = req.body;

    const isWin = process.platform === "win32";
    const venvPythonPath = isWin
      ? path.join(BASE_DIR, "VLCL_AI", ".venv", "Scripts", "python.exe")
      : path.join(BASE_DIR, "VLCL_AI", ".venv", "bin", "python3");
    const pythonCmd = fs.existsSync(venvPythonPath) ? venvPythonPath : (isWin ? "python" : "python3");

    const inlineScript = `
import sys, os, json, math
try:
    sys.path.insert(0, r"${BASE_DIR.replace(/\\/g, "\\\\")}")
    from VLCL_AI.environment.state import EnvironmentState
    from VLCL_AI.physics.physics_engine import PhysicsEngine
    from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    env_state = EnvironmentState(
        current_time=data.get("current_time", 0.0),
        frame_index=data.get("frame_index", 0),
        fps=data.get("fps", 60.0),
        receiver_position=data["receiver_position"],
        receiver_orientation=data["receiver_orientation"],
        receiver_velocity=data.get("receiver_velocity", [0,0,0]),
        receiver_acceleration=data.get("receiver_acceleration", [0,0,0]),
        receiver_angles=data.get("receiver_angles", {"roll":0,"pitch":0,"yaw":0}),
        room_dims=data.get("room_dims", [5.0, 5.0, 3.0]),
        led_positions={int(k): v for k,v in data["led_positions"].items()},
        led_powers={int(k): v for k,v in data["led_powers"].items()},
        led_active={int(k): v for k,v in data["led_active"].items()},
        led_orientations={int(k): v for k,v in data["led_orientations"].items()},
        led_beam_angles={int(k): v for k,v in data["led_beam_angles"].items()},
        distances={int(k): v for k,v in data["distances"].items()},
        incident_angles_rad={int(k): v * math.pi / 180.0 for k,v in data["incident_angles"].items()},
        irradiance_angles_rad={int(k): v * math.pi / 180.0 for k,v in data["irradiance_angles"].items()},
        visibility_matrix={int(k): v for k,v in data["visibility_matrix"].items()},
        los_matrix={int(k): v for k,v in data["los_matrix"].items()},
        blocking_obstacles={int(k): v for k,v in data["blocking_obstacles"].items()},
        obstacles=data.get("obstacles", [])
    )

    # Step 1: Run Physics Engine
    physics_engine = PhysicsEngine()
    physics_state = physics_engine.compute(env_state)

    # Step 2: Run Integrated Engine
    integrated_engine = IntegratedVLCLEngine()
    integrated_state = integrated_engine.step(env_state, physics_state)

    # Convert dict keys to str for JSON serialization
    result = integrated_state.to_dict()
    
    print(json.dumps(result))
except Exception as e:
    import traceback
    print(json.dumps({"__error__": str(e), "__traceback__": traceback.format_exc()}))
    sys.exit(1)
`;

    const tmpScript = path.join(BASE_DIR, "VLCL_AI", "logs", "_integrated_tmp.py");
    const tmpInput = path.join(BASE_DIR, "VLCL_AI", "logs", "_integrated_input.json");

    try {
      fs.mkdirSync(path.join(BASE_DIR, "VLCL_AI", "logs"), { recursive: true });
      fs.writeFileSync(tmpScript, inlineScript, "utf-8");
      fs.writeFileSync(tmpInput, JSON.stringify(envState), "utf-8");
    } catch (e: any) {
      return res.status(500).json({ error: "Failed to write temp files: " + e.message });
    }

    exec(
      `${pythonCmd} "${tmpScript}" "${tmpInput}"`,
      { cwd: BASE_DIR, env: { ...process.env, PYTHONIOENCODING: "utf-8", LOGURU_LEVEL: "WARNING" }, timeout: 35000 },
      (error, stdout, stderr) => {
        if (error && !stdout.trim()) {
          return res.status(500).json({ error: stderr || error.message });
        }
        try {
          const result = JSON.parse(stdout.trim());
          if (result.__error__) {
            return res.status(500).json({ error: result.__error__, traceback: result.__traceback__ });
          }
          res.json({ success: true, integrated: result });
        } catch {
          res.status(500).json({ error: "Failed to parse integrated output", raw: stdout.slice(0, 500) });
        }
      }
    );
  });

  // API Route: Adaptive Modulation & Dynamic Subcarrier Allocation (Module 6)
  // Chains Physics → AdaptiveTransmissionEngine → returns AllocationDecision summary
  app.post("/api/adaptive", (req, res) => {
    const envState = req.body;

    const isWin = process.platform === "win32";
    const venvPythonPath = isWin
      ? path.join(BASE_DIR, "VLCL_AI", ".venv", "Scripts", "python.exe")
      : path.join(BASE_DIR, "VLCL_AI", ".venv", "bin", "python3");
    const pythonCmd = fs.existsSync(venvPythonPath) ? venvPythonPath : (isWin ? "python" : "python3");

    const inlineScript = `
import sys, os, json, math, numpy as np
try:
    sys.path.insert(0, r"${BASE_DIR.replace(/\\/g, "\\\\")}")
    from VLCL_AI.environment.state import EnvironmentState
    from VLCL_AI.physics.physics_engine import PhysicsEngine
    from VLCL_AI.communication.subcarrier_grid import SubcarrierGrid
    from VLCL_AI.adaptive.engine import AdaptiveTransmissionEngine
    from VLCL_AI.adaptive.config import AdaptiveConfig
    from VLCL_AI.adaptive.feedback import ChannelFeedback

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    env_state = EnvironmentState(
        current_time=data.get("current_time", 0.0),
        frame_index=data.get("frame_index", 0),
        fps=data.get("fps", 60.0),
        receiver_position=data["receiver_position"],
        receiver_orientation=data["receiver_orientation"],
        receiver_velocity=data.get("receiver_velocity", [0,0,0]),
        receiver_acceleration=data.get("receiver_acceleration", [0,0,0]),
        receiver_angles=data.get("receiver_angles", {"roll":0,"pitch":0,"yaw":0}),
        room_dims=data.get("room_dims", [5.0, 5.0, 3.0]),
        led_positions={int(k): v for k,v in data["led_positions"].items()},
        led_powers={int(k): v for k,v in data["led_powers"].items()},
        led_active={int(k): v for k,v in data["led_active"].items()},
        led_orientations={int(k): v for k,v in data["led_orientations"].items()},
        led_beam_angles={int(k): v for k,v in data["led_beam_angles"].items()},
        distances={int(k): v for k,v in data["distances"].items()},
        incident_angles_rad={int(k): v * math.pi / 180.0 for k,v in data["incident_angles"].items()},
        irradiance_angles_rad={int(k): v * math.pi / 180.0 for k,v in data["irradiance_angles"].items()},
        visibility_matrix={int(k): v for k,v in data["visibility_matrix"].items()},
        los_matrix={int(k): v for k,v in data["los_matrix"].items()},
        blocking_obstacles={int(k): v for k,v in data["blocking_obstacles"].items()},
        obstacles=data.get("obstacles", [])
    )

    # Step 1: Run Physics Engine to get channel gains and SNR
    physics_engine = PhysicsEngine()
    physics_state = physics_engine.compute(env_state)

    # Step 2: Build AdaptiveConfig from request params
    num_devices = int(data.get("num_devices", len(data["led_positions"])))
    fft_size = int(data.get("fft_size", 256))
    ber_max = float(data.get("ber_max", 3.8e-3))
    mode = str(data.get("mode", "ADAPTIVE"))
    min_rate_bps = float(data.get("min_rate_bps", 5e6))

    config = AdaptiveConfig(
        ber_max=ber_max,
        mode=mode,
        fft_size=fft_size,
        total_bandwidth_hz=20.0e6,
        cp_ratio=0.25,
    )

    # Step 3: Build SubcarrierGrid
    grid = SubcarrierGrid(fft_size=fft_size, total_bandwidth=20.0e6)

    # Step 4: Build synthetic per-device, per-subcarrier SNR matrix
    # Use physics SNR (dB) → linear, broadcast across subcarriers with mild freq rolloff
    device_ids = list(range(1, num_devices + 1))
    snr_matrix = np.zeros((num_devices, fft_size), dtype=float)
    freq_axis = np.array([n * (20.0e6 / fft_size) for n in range(fft_size)])

    for k_idx, led_id in enumerate(device_ids):
        snr_db = physics_state.snrs.get(led_id, 0.0)
        snr_linear = 10.0 ** (snr_db / 10.0)
        # Mild 1st-order frequency rolloff: H(f) = 1 / (1 + (f/f_c)^2)
        f_cutoff = 20.0e6
        rolloff = 1.0 / (1.0 + (freq_axis / f_cutoff) ** 2)
        snr_matrix[k_idx, :] = snr_linear * rolloff
        snr_matrix[k_idx, 0] = 0.0  # DC subcarrier unused

    # Step 5: Build ChannelFeedback for each device
    feedbacks = [
        ChannelFeedback(
            device_id=dev_id,
            snr_per_subcarrier=snr_matrix[k_idx, :],
            requested_min_rate_bps=min_rate_bps,
        )
        for k_idx, dev_id in enumerate(device_ids)
    ]

    # Step 6: Run Module 6
    engine = AdaptiveTransmissionEngine(config=config)
    decision = engine.allocate_resources(feedbacks=feedbacks, grid=grid)

    result = decision.to_dict()
    print(json.dumps(result))
except Exception as e:
    import traceback
    print(json.dumps({"__error__": str(e), "__traceback__": traceback.format_exc()}))
    sys.exit(1)
`;

    const tmpScript = path.join(BASE_DIR, "VLCL_AI", "logs", "_adaptive_tmp.py");
    const tmpInput = path.join(BASE_DIR, "VLCL_AI", "logs", "_adaptive_input.json");

    try {
      fs.mkdirSync(path.join(BASE_DIR, "VLCL_AI", "logs"), { recursive: true });
      fs.writeFileSync(tmpScript, inlineScript, "utf-8");
      fs.writeFileSync(tmpInput, JSON.stringify(envState), "utf-8");
    } catch (e: any) {
      return res.status(500).json({ error: "Failed to write temp files: " + e.message });
    }

    exec(
      `${pythonCmd} "${tmpScript}" "${tmpInput}"`,
      { cwd: BASE_DIR, env: { ...process.env, PYTHONIOENCODING: "utf-8", LOGURU_LEVEL: "WARNING" }, timeout: 35000 },
      (error, stdout, stderr) => {
        if (error && !stdout.trim()) {
          return res.status(500).json({ error: stderr || error.message });
        }
        try {
          const result = JSON.parse(stdout.trim());
          if (result.__error__) {
            return res.status(500).json({ error: result.__error__, traceback: result.__traceback__ });
          }
          res.json({ success: true, adaptive: result });
        } catch {
          res.status(500).json({ error: "Failed to parse adaptive output", raw: stdout.slice(0, 500) });
        }
      }
    );
  });


  // API Route: Power Allocation & LED Pre-Equalization (Module 7)
  // Chains Physics → Module 6 → PowerPreEqualizationEngine → returns PowerDecision summary
  app.post("/api/power", (req, res) => {
    const envState = req.body;

    const isWin = process.platform === "win32";
    const venvPythonPath = isWin
      ? path.join(BASE_DIR, "VLCL_AI", ".venv", "Scripts", "python.exe")
      : path.join(BASE_DIR, "VLCL_AI", ".venv", "bin", "python3");
    const pythonCmd = fs.existsSync(venvPythonPath) ? venvPythonPath : (isWin ? "python" : "python3");

    const inlineScript = `
import sys, os, json, math, numpy as np
try:
    sys.path.insert(0, r"${BASE_DIR.replace(/\\/g, "\\\\")}")
    from VLCL_AI.environment.state import EnvironmentState
    from VLCL_AI.physics.physics_engine import PhysicsEngine
    from VLCL_AI.communication.subcarrier_grid import SubcarrierGrid
    from VLCL_AI.adaptive.engine import AdaptiveTransmissionEngine
    from VLCL_AI.adaptive.power_engine import PowerPreEqualizationEngine
    from VLCL_AI.adaptive.config import AdaptiveConfig
    from VLCL_AI.adaptive.feedback import ChannelFeedback

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    env_state = EnvironmentState(
        current_time=data.get("current_time", 0.0),
        frame_index=data.get("frame_index", 0),
        fps=data.get("fps", 60.0),
        receiver_position=data["receiver_position"],
        receiver_orientation=data["receiver_orientation"],
        receiver_velocity=data.get("receiver_velocity", [0,0,0]),
        receiver_acceleration=data.get("receiver_acceleration", [0,0,0]),
        receiver_angles=data.get("receiver_angles", {"roll":0,"pitch":0,"yaw":0}),
        room_dims=data.get("room_dims", [5.0, 5.0, 3.0]),
        led_positions={int(k): v for k,v in data["led_positions"].items()},
        led_powers={int(k): v for k,v in data["led_powers"].items()},
        led_active={int(k): v for k,v in data["led_active"].items()},
        led_orientations={int(k): v for k,v in data["led_orientations"].items()},
        led_beam_angles={int(k): v for k,v in data["led_beam_angles"].items()},
        distances={int(k): v for k,v in data["distances"].items()},
        incident_angles_rad={int(k): v * math.pi / 180.0 for k,v in data["incident_angles"].items()},
        irradiance_angles_rad={int(k): v * math.pi / 180.0 for k,v in data["irradiance_angles"].items()},
        visibility_matrix={int(k): v for k,v in data["visibility_matrix"].items()},
        los_matrix={int(k): v for k,v in data["los_matrix"].items()},
        blocking_obstacles={int(k): v for k,v in data["blocking_obstacles"].items()},
        obstacles=data.get("obstacles", [])
    )

    # Step 1: Physics Engine
    physics_engine = PhysicsEngine()
    physics_state = physics_engine.compute(env_state)

    # Step 2: Build config and grid
    num_devices = int(data.get("num_devices", len(data["led_positions"])))
    fft_size = int(data.get("fft_size", 256))
    ber_max = float(data.get("ber_max", 3.8e-3))
    mode = str(data.get("mode", "ADAPTIVE"))
    min_rate_bps = float(data.get("min_rate_bps", 5e6))

    config = AdaptiveConfig(ber_max=ber_max, mode=mode, fft_size=fft_size, total_bandwidth_hz=20.0e6)
    grid = SubcarrierGrid(fft_size=fft_size, total_bandwidth=20.0e6)

    from VLCL_AI.communication.snr import compute_communication_snr
    from VLCL_AI.communication.led_frequency_response import LEDFrequencyResponse
    from VLCL_AI.physics.constants import DEFAULT_RESPONSIVITY

    # Step 3: Build per-subcarrier SNR matrix using actual optical channel model
    # BUG FIX: was using physics_state.snrs (single broadband scalar) with an overly
    # aggressive rolloff. Now uses compute_communication_snr with actual optical gains
    # and the LED frequency response model — consistent with how power_engine.py computes BER.
    device_ids = list(range(1, num_devices + 1))
    snr_matrix = np.zeros((num_devices, fft_size), dtype=float)
    freq_axis = np.array([n * grid.subcarrier_spacing for n in range(fft_size)])
    led_response = LEDFrequencyResponse(model_type="first_order", cutoff_frequency_hz=10.0e6)

    for k_idx, led_id in enumerate(device_ids):
        opt_gain = physics_state.total_gains.get(led_id, 1e-6)
        noise_var = physics_state.noise_variances.get(led_id, 1e-12)
        # Build per-subcarrier channel gains including LED frequency rolloff
        channel_gains = np.zeros((num_devices, fft_size), dtype=float)
        for n in range(fft_size):
            channel_gains[k_idx, n] = opt_gain * led_response.magnitude(freq_axis[n])
        # 1 W seed power for SNR estimation (same as BOOTSTRAP_POWER_W in integrated engine)
        seed_powers = np.ones((fft_size, num_devices), dtype=float)
        snr_k = compute_communication_snr(
            responsivity=DEFAULT_RESPONSIVITY,
            subcarrier_powers=seed_powers,
            channel_gains=channel_gains,
            noise_variance=noise_var
        )
        snr_k[0] = 0.0  # Zero DC subcarrier
        snr_matrix[k_idx, :] = snr_k

    # Step 4: Run Module 6
    # BUG FIX: was using list comprehension with closure over k_idx — the comprehension
    # captured the final value of k_idx (= num_devices-1), giving every device the same
    # (last) SNR row. Now uses explicit indexing.
    feedbacks = [
        ChannelFeedback(device_id=dev_id, snr_per_subcarrier=snr_matrix[idx, :], requested_min_rate_bps=min_rate_bps)
        for idx, dev_id in enumerate(device_ids)
    ]
    adaptive_engine = AdaptiveTransmissionEngine(config=config)
    decision = adaptive_engine.allocate_resources(feedbacks=feedbacks, grid=grid)

    # Step 5: Run Module 7
    power_engine = PowerPreEqualizationEngine(config=config)
    total_power_budget_w = float(data.get("total_power_budget_w", 4.0))
    power_mode = str(data.get("power_mode", "EQUAL_POWER"))
    pre_eq_mode = str(data.get("pre_eq_mode", "REGULARIZED"))
    loc_reserve_w = float(data.get("localization_reserve_w", 0.1))

    power_decision = power_engine.process_power_and_preeq(
        allocation_decision=decision,
        physics_state=physics_state,
        grid=grid,
        total_power_budget_w=total_power_budget_w,
        localization_reserve_w=loc_reserve_w,
        power_mode=power_mode,
        pre_eq_mode=pre_eq_mode,
    )

    pa = power_decision.power_allocation
    pe = power_decision.pre_eq_state

    result = {
        "power_allocation": {
            "mode": pa.mode,
            "total_power_budget_w": float(pa.total_power_budget_w),
            "per_led_max_power_w": {str(k): float(v) for k, v in pa.per_led_max_power_w.items()},
            "localization_reserved_power_w": {str(k): float(v) for k, v in pa.localization_reserved_power_w.items()},
            "communication_available_power_w": {str(k): float(v) for k, v in pa.communication_available_power_w.items()},
            "per_device_power_w": {str(k): float(v) for k, v in pa.per_device_power_w.items()},
        },
        "pre_eq": {
            "mode": pe.mode,
            "max_gain_db": float(pe.max_gain_db),
            "papr_before_db": {str(k): float(v) for k, v in pe.papr_before_db.items()},
            "papr_after_db": {str(k): float(v) for k, v in pe.papr_after_db.items()},
            "clipping_ratio": {str(k): float(v) for k, v in pe.clipping_ratio.items()},
        },
        "predicted_ber": {str(k): float(v) for k, v in power_decision.predicted_ber.items()},
        "modulation_feasible": {str(k): bool(v) for k, v in power_decision.modulation_feasible.items()},
        "nominal_sum_rate_bps": float(power_decision.nominal_sum_rate_bps),
        "feasible_sum_rate_bps": float(power_decision.feasible_sum_rate_bps),
        "warnings": list(power_decision.warnings),
    }
    print(json.dumps(result))
except Exception as e:
    import traceback
    print(json.dumps({"__error__": str(e), "__traceback__": traceback.format_exc()}))
    sys.exit(1)
`;

    const tmpScript = path.join(BASE_DIR, "VLCL_AI", "logs", "_power_tmp.py");
    const tmpInput = path.join(BASE_DIR, "VLCL_AI", "logs", "_power_input.json");

    try {
      fs.mkdirSync(path.join(BASE_DIR, "VLCL_AI", "logs"), { recursive: true });
      fs.writeFileSync(tmpScript, inlineScript, "utf-8");
      fs.writeFileSync(tmpInput, JSON.stringify(envState), "utf-8");
    } catch (e: any) {
      return res.status(500).json({ error: "Failed to write temp files: " + e.message });
    }

    exec(
      `${pythonCmd} "${tmpScript}" "${tmpInput}"`,
      { cwd: BASE_DIR, env: { ...process.env, PYTHONIOENCODING: "utf-8", LOGURU_LEVEL: "WARNING" }, timeout: 40000 },
      (error, stdout, stderr) => {
        if (error && !stdout.trim()) {
          return res.status(500).json({ error: stderr || error.message });
        }
        try {
          const outStr = stdout.trim();
          const jsonStr = outStr.substring(outStr.indexOf('{'));
          const result = JSON.parse(jsonStr);
          if (result.__error__) {
            return res.status(500).json({ error: result.__error__, traceback: result.__traceback__ });
          }
          res.json({ success: true, power: result });
        } catch {
          res.status(500).json({ error: "Failed to parse power output", raw: stdout.slice(0, 500) });
        }
      }
    );
  });


  // API Route: Joint Optimization (Module 8)
  // Chains Physics → JointAdaptiveOptimizer → returns JointDecisionState summary
  app.post("/api/joint", (req, res) => {
    const envState = req.body;

    const isWin = process.platform === "win32";
    const venvPythonPath = isWin
      ? path.join(BASE_DIR, "VLCL_AI", ".venv", "Scripts", "python.exe")
      : path.join(BASE_DIR, "VLCL_AI", ".venv", "bin", "python3");
    const pythonCmd = fs.existsSync(venvPythonPath) ? venvPythonPath : (isWin ? "python" : "python3");

    const inlineScript = `
import sys, os, json, math, numpy as np
try:
    sys.path.insert(0, r"${BASE_DIR.replace(/\\/g, "\\\\")}")
    from VLCL_AI.environment.state import EnvironmentState
    from VLCL_AI.physics.physics_engine import PhysicsEngine
    from VLCL_AI.adaptive.joint_optimizer import JointAdaptiveOptimizer
    from VLCL_AI.adaptive.config import AdaptiveConfig

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    env_state = EnvironmentState(
        current_time=data.get("current_time", 0.0),
        frame_index=data.get("frame_index", 0),
        fps=data.get("fps", 60.0),
        receiver_position=data["receiver_position"],
        receiver_orientation=data["receiver_orientation"],
        receiver_velocity=data.get("receiver_velocity", [0,0,0]),
        receiver_acceleration=data.get("receiver_acceleration", [0,0,0]),
        receiver_angles=data.get("receiver_angles", {"roll":0,"pitch":0,"yaw":0}),
        room_dims=data.get("room_dims", [5.0, 5.0, 3.0]),
        led_positions={int(k): v for k,v in data["led_positions"].items()},
        led_powers={int(k): v for k,v in data["led_powers"].items()},
        led_active={int(k): v for k,v in data["led_active"].items()},
        led_orientations={int(k): v for k,v in data["led_orientations"].items()},
        led_beam_angles={int(k): v for k,v in data["led_beam_angles"].items()},
        distances={int(k): v for k,v in data["distances"].items()},
        incident_angles_rad={int(k): v * math.pi / 180.0 for k,v in data["incident_angles"].items()},
        irradiance_angles_rad={int(k): v * math.pi / 180.0 for k,v in data["irradiance_angles"].items()},
        visibility_matrix={int(k): v for k,v in data["visibility_matrix"].items()},
        los_matrix={int(k): v for k,v in data["los_matrix"].items()},
        blocking_obstacles={int(k): v for k,v in data["blocking_obstacles"].items()},
        obstacles=data.get("obstacles", [])
    )

    # Step 1: Physics Engine
    physics_engine = PhysicsEngine()
    physics_state = physics_engine.compute(env_state)

    # Extract params
    num_devices = int(data.get("num_devices", len(data["led_positions"])))
    min_rate_bps = float(data.get("min_rate_bps", 5e6))
    ber_max = float(data.get("ber_max", 3.8e-3))
    total_power_budget_w = float(data.get("total_power_budget_w", 40.0))
    per_led_max_power_w = float(data.get("per_led_max_power_w", 10.0))
    target_loc_error_m = float(data.get("target_localization_error_m", 0.20))
    power_mode = str(data.get("power_mode", "WATER_FILLING"))
    pre_eq_mode = str(data.get("pre_eq_mode", "REGULARIZED"))
    max_iterations = int(data.get("max_iterations", 6))

    config = AdaptiveConfig(
        ber_max=ber_max,
        mode="ADAPTIVE",
        fft_size=256,
        total_bandwidth_hz=20.0e6,
        cp_ratio=0.25,
    )

    optimizer = JointAdaptiveOptimizer(
        config=config,
        target_localization_error_m=target_loc_error_m,
        ber_max=ber_max,
        total_power_budget_w=total_power_budget_w,
        per_led_max_power_w=per_led_max_power_w,
        max_iterations=max_iterations
    )

    min_rates = {k: min_rate_bps for k in range(1, num_devices + 1)}

    joint_state = optimizer.optimize(
        env_state=env_state,
        physics_state=physics_state,
        min_rates_bps=min_rates,
        bits_dict=None,
        power_mode=power_mode,
        pre_eq_mode=pre_eq_mode
    )

    result = joint_state.to_dict()
    print(json.dumps(result))
except Exception as e:
    import traceback
    print(json.dumps({"__error__": str(e), "__traceback__": traceback.format_exc()}))
    sys.exit(1)
`;

    const tmpScript = path.join(BASE_DIR, "VLCL_AI", "logs", "_joint_tmp.py");
    const tmpInput = path.join(BASE_DIR, "VLCL_AI", "logs", "_joint_input.json");

    try {
      fs.mkdirSync(path.join(BASE_DIR, "VLCL_AI", "logs"), { recursive: true });
      fs.writeFileSync(tmpScript, inlineScript, "utf-8");
      fs.writeFileSync(tmpInput, JSON.stringify(envState), "utf-8");
    } catch (e: any) {
      return res.status(500).json({ error: "Failed to write temp files: " + e.message });
    }

    // Larger timeout since iterative optimization can take several seconds
    exec(
      `${pythonCmd} "${tmpScript}" "${tmpInput}"`,
      { cwd: BASE_DIR, env: { ...process.env, PYTHONIOENCODING: "utf-8", LOGURU_LEVEL: "WARNING" }, timeout: 60000 },
      (error, stdout, stderr) => {
        if (error && !stdout.trim()) {
          return res.status(500).json({ error: stderr || error.message });
        }
        try {
          const result = JSON.parse(stdout.trim());
          if (result.__error__) {
            return res.status(500).json({ error: result.__error__, traceback: result.__traceback__ });
          }
          res.json({ success: true, joint: result });
        } catch {
          res.status(500).json({ error: "Failed to parse joint output", raw: stdout.slice(0, 500) });
        }
      }
    );
  });


  app.get("/api/visualization", (req, res) => {
    const htmlPath = path.join(BASE_DIR, "VLCL_AI", "logs", "simulation_3d.html");
    if (fs.existsSync(htmlPath)) {
      res.sendFile(htmlPath);
    } else {
      res.status(404).send("Visualization not found. Run the simulation first.");
    }
  });

  app.get("/api/export-zip", (req, res) => {
    const zipPath = path.join(BASE_DIR, "dist", "VLCL_AI_Module1.zip");
    fs.mkdirSync(path.dirname(zipPath), { recursive: true });

    // Use the system zip command
    exec(`zip -r "${zipPath}" VLCL_AI`, { cwd: BASE_DIR }, (error, stdout, stderr) => {
      if (error) {
        return res.status(500).json({ error: "Failed to zip package: " + error.message });
      }
      res.download(zipPath, "VLCL_AI_Module1.zip");
    });
  });

  app.get("/api/files", (req, res) => {
    const basePath = path.join(BASE_DIR, "VLCL_AI");
    const filesList: any[] = [];

    function scanDir(dir: string) {
      if (!fs.existsSync(dir)) return;
      const items = fs.readdirSync(dir);
      for (const item of items) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);
        const relPath = path.relative(basePath, fullPath);
        if (stat.isDirectory()) {
          if (item !== "__pycache__" && item !== "logs" && item !== "assets") {
            scanDir(fullPath);
          }
        } else {
          // Read up to 2MB text
          filesList.push({
            path: relPath,
            name: item,
            content: fs.readFileSync(fullPath, "utf-8")
          });
        }
      }
    }

    try {
      scanDir(basePath);
      res.json(filesList);
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  // API Route: Get dataset configuration
  app.get("/api/dataset/config", (req, res) => {
    const configPath = path.join(BASE_DIR, "VLCL_AI", "configs", "dataset_config.yaml");
    if (fs.existsSync(configPath)) {
      res.send(fs.readFileSync(configPath, "utf-8"));
    } else {
      res.status(404).json({ error: "Dataset config not found." });
    }
  });

  // API Route: Save dataset configuration
  app.post("/api/dataset/config", (req, res) => {
    const { content } = req.body;
    const configPath = path.join(BASE_DIR, "VLCL_AI", "configs", "dataset_config.yaml");
    try {
      fs.mkdirSync(path.dirname(configPath), { recursive: true });
      fs.writeFileSync(configPath, content, "utf-8");
      res.json({ success: true });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  // API Route: Run dataset generation as SSE stream
  app.get("/api/dataset/generate-stream", (req, res) => {
    const scriptPath = path.join(BASE_DIR, "VLCL_AI", "examples", "run_dataset.py");
    const isWin = process.platform === "win32";
    const venvPythonPath = isWin
      ? path.join(BASE_DIR, "VLCL_AI", ".venv", "Scripts", "python.exe")
      : path.join(BASE_DIR, "VLCL_AI", ".venv", "bin", "python3");
    const pythonCmd = fs.existsSync(venvPythonPath) ? venvPythonPath : (isWin ? "python" : "python3");

    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive"
    });

    const child = spawn(pythonCmd, [scriptPath], {
      cwd: BASE_DIR,
      env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONPATH: BASE_DIR }
    });

    child.stdout.on("data", (data) => {
      res.write(`data: ${JSON.stringify({ type: "stdout", text: data.toString() })}\n\n`);
    });

    child.stderr.on("data", (data) => {
      res.write(`data: ${JSON.stringify({ type: "stderr", text: data.toString() })}\n\n`);
    });

    child.on("close", (code) => {
      res.write(`data: ${JSON.stringify({ type: "close", code })}\n\n`);
      res.end();
    });
    
    req.on("close", () => {
      child.kill();
    });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: process.env.DISABLE_HMR === "true" ? false : undefined
      },
      appType: "spa",
      root: path.join(BASE_DIR, "../frontend"),
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(BASE_DIR, "../frontend/dist");
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on port ${PORT}`);
  });
}

startServer();
