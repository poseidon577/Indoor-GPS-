import { useState, useEffect, useRef } from "react";
import { Terminal, Layers, RefreshCw, FileText, Download, Activity, BookOpen, Lightbulb, MapPin, Table2, Database } from "lucide-react";

import { SimulationState, RoomParams, LEDParams, ReceiverParams, ObstacleParams, MobilityParams } from "./types";
import ThreeCanvas from "./components/ThreeCanvas";
import ControlPanel from "./components/ControlPanel";
import DebugOverlay from "./components/DebugOverlay";
import FormulaPanel from "./components/FormulaPanel";
import CodeViewer from "./components/CodeViewer";
import IllustrationPanel from "./components/IllustrationPanel";
import LiveResults from "./components/results/LiveResults";
import { useLiveResults } from "./components/results/useLiveResults";
import DatasetPanel from "./components/DatasetPanel";

// ─── 1. Initial State Definitions ─────────────────────────────────────────
const INITIAL_ROOM: RoomParams = {
  width: 5.0,
  length: 5.0,
  height: 3.0,
  wallReflectivity: 0.8,
  floorReflectivity: 0.2,
  ceilingReflectivity: 0.5
};

const INITIAL_LEDS: LEDParams[] = [
  { id: 1, position: [1.25, 1.25, 3.0], orientation: [0, 0, -1], power: 20.0, biasCurrent: 0.5, frequency: 100000.0, lambertianOrder: 1.0, beamAngle: 60.0, fov: 60.0, communicationEnabled: true, localizationEnabled: true },
  { id: 2, position: [3.75, 1.25, 3.0], orientation: [0, 0, -1], power: 20.0, biasCurrent: 0.5, frequency: 150000.0, lambertianOrder: 1.0, beamAngle: 60.0, fov: 60.0, communicationEnabled: true, localizationEnabled: true },
  { id: 3, position: [1.25, 3.75, 3.0], orientation: [0, 0, -1], power: 20.0, biasCurrent: 0.5, frequency: 200000.0, lambertianOrder: 1.0, beamAngle: 60.0, fov: 60.0, communicationEnabled: true, localizationEnabled: true },
  { id: 4, position: [3.75, 3.75, 3.0], orientation: [0, 0, -1], power: 20.0, biasCurrent: 0.5, frequency: 250000.0, lambertianOrder: 1.0, beamAngle: 60.0, fov: 60.0, communicationEnabled: true, localizationEnabled: true },
];

const INITIAL_RECEIVER: ReceiverParams = {
  position: [2.5, 2.5, 0.85],
  orientation: [0, 0, 1],
  velocity: [0.2, 0.1, 0.0],
  acceleration: [0.0, 0.0, 0.0],
  fov: 70.0,
  apdSize: 1e-4,
  noise: 1e-14,
  gain: 1.0,
  roll: 0.0,
  pitch: 0.0,
  yaw: 0.0
};

const INITIAL_OBSTACLES: ObstacleParams[] = [
  { id: "obs_human", type: "cylinder", position: [2.0, 3.0, 0.9], rotation: [0, 0, 0], scale: [0.3, 0.3, 1.8], reflectivity: 0.3, material: "skin_fabric" },
  { id: "obs_desk", type: "box", position: [3.5, 2.0, 0.4], rotation: [0, 0, 0], scale: [1.2, 0.8, 0.8], reflectivity: 0.4, material: "wood" },
];

const INITIAL_MOBILITY: MobilityParams = {
  type: "circular",
  speed: 0.5,
  radius: 1.5,
  center: [2.5, 2.5, 0.85],
  waypoints: [[1.0, 1.0, 0.85], [4.0, 1.0, 0.85], [4.0, 4.0, 0.85], [1.0, 4.0, 0.85]]
};

// ─── Tab Config ────────────────────────────────────────────────────────────
type TabKey = "visualizer" | "results" | "dataset" | "guide" | "formulas" | "code" | "terminal";

const TABS: { key: TabKey; icon: React.ReactNode; label: string; desc: string }[] = [
  { key: "visualizer", icon: <Layers className="w-3.5 h-3.5" />, label: "3D Simulator", desc: "Live 3D digital twin" },
  { key: "results", icon: <Table2 className="w-3.5 h-3.5" />, label: "Live Results", desc: "Real-time scientific workbook" },
  { key: "dataset", icon: <Database className="w-3.5 h-3.5" />, label: "Data Factory", desc: "Generate synthetic datasets" },
  { key: "guide", icon: <Lightbulb className="w-3.5 h-3.5" />, label: "System Guide", desc: "How it works — for beginners" },
  { key: "formulas", icon: <BookOpen className="w-3.5 h-3.5" />, label: "Physics Math", desc: "Formulas & diagrams" },
  { key: "code", icon: <FileText className="w-3.5 h-3.5" />, label: "Codebase", desc: "View source code" },
  { key: "terminal", icon: <Terminal className="w-3.5 h-3.5" />, label: "Python Terminal", desc: "Run simulation engine" },
];

// ─── 3D View Legend ────────────────────────────────────────────────────────
function ViewLegend() {
  const items = [
    { color: "bg-white", label: "LOS Ray — clear path" },
    { color: "bg-red-500", label: "NLOS Ray — blocked" },
    { color: "bg-yellow-400", label: "LED Comm cone" },
    { color: "bg-amber-200", label: "LED Loc cone" },
    { color: "bg-cyan-400", label: "Receiver FOV" },
    { color: "bg-red-700", label: "Obstacle" },
    { color: "bg-orange-400", label: "Estimated Position (A-DPDOA)" },
  ];
  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 backdrop-blur-sm">
      <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-1.5">
        <Activity className="w-3 h-3 text-cyan-400" /> 3D View Legend
      </h4>
      <div className="flex flex-col gap-2">
        {items.map((item, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className={`w-3 h-1.5 rounded-sm ${item.color} flex-shrink-0`} />
            <span className="text-slate-400 text-[11px]">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Main App ──────────────────────────────────────────────────────────────
export default function App() {
  const [state, setState] = useState<SimulationState>({
    currentTime: 0.0,
    frameIndex: 0,
    fps: 60.0,
    isPlaying: true,
    speedFactor: 1.0,
    room: INITIAL_ROOM,
    leds: INITIAL_LEDS,
    receiver: INITIAL_RECEIVER,
    obstacles: INITIAL_OBSTACLES,
    mobility: INITIAL_MOBILITY,
    distances: {},
    incidentAngles: {},
    irradianceAngles: {},
    dcGains: {},
    losMatrix: {},
    visibilityMatrix: {},
    blockingObstacles: {},
    trajectoryPoints: [[2.5, 2.5, 0.85]],
    physicsMetrics: null,
    physicsLoading: false,
    commMetrics: null,
    commLoading: false,
    localizationMetrics: null,
    localizationLoading: false,
    integratedMetrics: null,
    integratedLoading: false,
    adaptiveMetrics: null,
    adaptiveLoading: false,
    powerMetrics: null,
    powerLoading: false,
    jointMetrics: null,
    jointLoading: false,
  });

  const [activeTab, setActiveTab] = useState<TabKey>("visualizer");
  const [pythonTerminalLogs, setPythonTerminalLogs] = useState<string>("");
  const [pythonLoading, setPythonLoading] = useState(false);
  const [pythonSuccess, setPythonSuccess] = useState<boolean | null>(null);
  const liveResults = useLiveResults(state);

  const timeElapsedRef = useRef(0.0);
  const waypointIndexRef = useRef(0);
  const randomWalkTimerRef = useRef(0.0);
  const randomWalkDirRef = useRef<[number, number, number]>([1, 0, 0]);

  const stateRef = useRef(state);
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  // ─── 2. Helpers ──────────────────────────────────────────────────────────
  const getRotationMatrix = (rollDeg: number, pitchDeg: number, yawDeg: number) => {
    const r = (rollDeg * Math.PI) / 180;
    const p = (pitchDeg * Math.PI) / 180;
    const y = (yawDeg * Math.PI) / 180;
    const cosR = Math.cos(r), sinR = Math.sin(r);
    const cosP = Math.cos(p), sinP = Math.sin(p);
    const cosY = Math.cos(y), sinY = Math.sin(y);
    const Rx = [[1, 0, 0], [0, cosR, -sinR], [0, sinR, cosR]];
    const Ry = [[cosP, 0, sinP], [0, 1, 0], [-sinP, 0, cosP]];
    const Rz = [[cosY, -sinY, 0], [sinY, cosY, 0], [0, 0, 1]];
    const mul = (A: number[][], B: number[][]) => {
      const C = Array(3).fill(0).map(() => Array(3).fill(0));
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
          for (let k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
      return C;
    };
    return mul(Rz, mul(Ry, Rx));
  };

  const getRotatedNormal = (roll: number, pitch: number, yaw: number): [number, number, number] => {
    const R = getRotationMatrix(roll, pitch, yaw);
    return [R[0][2], R[1][2], R[2][2]];
  };

  const rayCylinderIntersect = (
    pTx: [number, number, number], rayDir: [number, number, number],
    cylPos: [number, number, number], radius: number, height: number, rayDist: number
  ): { hit: boolean; t: number } => {
    const [ox, oy] = pTx, [dx, dy] = rayDir, [cx, cy] = cylPos;
    const A = dx * dx + dy * dy;
    if (Math.abs(A) < 1e-8) {
      const dSq = (ox - cx) ** 2 + (oy - cy) ** 2;
      if (dSq <= radius * radius) {
        const t = (cylPos[2] - height / 2 - pTx[2]) / rayDir[2];
        if (t > 0 && t < rayDist) return { hit: true, t };
      }
      return { hit: false, t: Infinity };
    }
    const B = 2 * (dx * (ox - cx) + dy * (oy - cy));
    const C = (ox - cx) ** 2 + (oy - cy) ** 2 - radius ** 2;
    const disc = B * B - 4 * A * C;
    if (disc < 0) return { hit: false, t: Infinity };
    for (const t of [(-B - Math.sqrt(disc)) / (2 * A), (-B + Math.sqrt(disc)) / (2 * A)]) {
      if (t > 0.01 && t < rayDist - 0.01) {
        const z = pTx[2] + t * rayDir[2];
        if (z >= cylPos[2] - height / 2 && z <= cylPos[2] + height / 2) return { hit: true, t };
      }
    }
    return { hit: false, t: Infinity };
  };

  const rayBoxIntersect = (
    pTx: [number, number, number], rayDir: [number, number, number],
    boxPos: [number, number, number], scale: [number, number, number], rayDist: number
  ): { hit: boolean; t: number } => {
    const bMin = [boxPos[0] - scale[0] / 2, boxPos[1] - scale[1] / 2, boxPos[2] - scale[2] / 2];
    const bMax = [boxPos[0] + scale[0] / 2, boxPos[1] + scale[1] / 2, boxPos[2] + scale[2] / 2];
    let tMin = -Infinity, tMax = Infinity;
    for (let i = 0; i < 3; i++) {
      if (Math.abs(rayDir[i]) < 1e-8) {
        if (pTx[i] < bMin[i] || pTx[i] > bMax[i]) return { hit: false, t: Infinity };
      } else {
        const t1 = (bMin[i] - pTx[i]) / rayDir[i], t2 = (bMax[i] - pTx[i]) / rayDir[i];
        tMin = Math.max(tMin, Math.min(t1, t2));
        tMax = Math.min(tMax, Math.max(t1, t2));
      }
    }
    if (tMax >= tMin && tMax > 0.01) {
      const t = tMin > 0.01 ? tMin : tMax;
      if (t < rayDist - 0.01) return { hit: true, t };
    }
    return { hit: false, t: Infinity };
  };

  // ─── 3. Physics Loop ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!state.isPlaying) return;
    const dt = 0.05 * state.speedFactor;
    const timer = setInterval(() => {
      timeElapsedRef.current += dt;
      const currentState = stateRef.current;
      let [rxX, rxY, rxZ] = currentState.receiver.position;
      let [vx, vy, vz] = currentState.receiver.velocity;

      if (currentState.mobility.type === "circular") {
        const omega = currentState.mobility.speed / currentState.mobility.radius;
        const theta = omega * timeElapsedRef.current;
        rxX = currentState.mobility.center[0] + currentState.mobility.radius * Math.cos(theta);
        rxY = currentState.mobility.center[1] + currentState.mobility.radius * Math.sin(theta);
        rxZ = currentState.mobility.center[2];
        vx = -currentState.mobility.speed * Math.sin(theta);
        vy = currentState.mobility.speed * Math.cos(theta);
        vz = 0;
      } else if (currentState.mobility.type === "linear") {
        rxX += vx * dt; rxY += vy * dt;
        if (rxX < 0.1 || rxX > currentState.room.width - 0.1) { rxX = Math.max(0.1, Math.min(currentState.room.width - 0.1, rxX)); vx = -vx; }
        if (rxY < 0.1 || rxY > currentState.room.length - 0.1) { rxY = Math.max(0.1, Math.min(currentState.room.length - 0.1, rxY)); vy = -vy; }
      } else if (currentState.mobility.type === "random_walk") {
        randomWalkTimerRef.current += dt;
        if (randomWalkTimerRef.current > 2.0 || (vx === 0 && vy === 0)) {
          randomWalkTimerRef.current = 0;
          const angle = Math.random() * 2 * Math.PI;
          randomWalkDirRef.current = [Math.cos(angle), Math.sin(angle), 0];
          vx = randomWalkDirRef.current[0] * currentState.mobility.speed;
          vy = randomWalkDirRef.current[1] * currentState.mobility.speed;
        }
        rxX += vx * dt; rxY += vy * dt;
        if (rxX < 0.1 || rxX > currentState.room.width - 0.1) { rxX = Math.max(0.1, Math.min(currentState.room.width - 0.1, rxX)); vx = -vx; }
        if (rxY < 0.1 || rxY > currentState.room.length - 0.1) { rxY = Math.max(0.1, Math.min(currentState.room.length - 0.1, rxY)); vy = -vy; }
      } else if (currentState.mobility.type === "waypoint" && currentState.mobility.waypoints.length > 0) {
        const target = currentState.mobility.waypoints[waypointIndexRef.current];
        const dx = target[0] - rxX, dy = target[1] - rxY, dz = target[2] - rxZ;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < 0.15) {
          waypointIndexRef.current = (waypointIndexRef.current + 1) % currentState.mobility.waypoints.length;
        } else {
          vx = (dx / dist) * currentState.mobility.speed; vy = (dy / dist) * currentState.mobility.speed; vz = (dz / dist) * currentState.mobility.speed;
          rxX += vx * dt; rxY += vy * dt; rxZ += vz * dt;
        }
      }

      const nextRxPos: [number, number, number] = [rxX, rxY, rxZ];
      const rxNormal = getRotatedNormal(currentState.receiver.roll, currentState.receiver.pitch, currentState.receiver.yaw);

      const distances: Record<number, number> = {};
      const irradianceAngles: Record<number, number> = {};
      const incidentAngles: Record<number, number> = {};
      const losMatrix: Record<number, boolean> = {};
      const visibilityMatrix: Record<number, boolean> = {};
      const blockingObstacles: Record<number, string> = {};
      const dcGains: Record<number, number> = {};

      currentState.leds.forEach((led) => {
        const dx = nextRxPos[0] - led.position[0], dy = nextRxPos[1] - led.position[1], dz = nextRxPos[2] - led.position[2];
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        distances[led.id] = d;
        if (d === 0) { irradianceAngles[led.id] = 0; incidentAngles[led.id] = 0; return; }

        const vTr: [number, number, number] = [dx / d, dy / d, dz / d];
        const cosPhi = -(vTr[2]);
        irradianceAngles[led.id] = (Math.acos(Math.max(-1, Math.min(1, cosPhi))) * 180) / Math.PI;
        const cosPsi = -(vTr[0] * rxNormal[0] + vTr[1] * rxNormal[1] + vTr[2] * rxNormal[2]);
        incidentAngles[led.id] = (Math.acos(Math.max(-1, Math.min(1, cosPsi))) * 180) / Math.PI;

        let isLos = true, blockingObsId = "";
        for (const obs of currentState.obstacles) {
          if (obs.type === "cylinder") {
            const { hit } = rayCylinderIntersect(led.position, vTr, obs.position, obs.scale[0], obs.scale[2], d);
            if (hit) { isLos = false; blockingObsId = obs.id; break; }
          } else if (obs.type === "box") {
            const { hit } = rayBoxIntersect(led.position, vTr, obs.position, obs.scale, d);
            if (hit) { isLos = false; blockingObsId = obs.id; break; }
          }
        }
        losMatrix[led.id] = isLos;
        blockingObstacles[led.id] = blockingObsId;

        const inFov = Math.abs(incidentAngles[led.id]) <= currentState.receiver.fov && Math.abs(irradianceAngles[led.id]) <= led.fov;
        visibilityMatrix[led.id] = inFov;

        if (isLos && inFov && cosPhi > 0 && cosPsi > 0) {
          const radHalf = (led.beamAngle * Math.PI) / 180;
          const m = -Math.log(2.0) / Math.log(Math.cos(radHalf / 2));
          const coeff = ((m + 1) * currentState.receiver.apdSize) / (2.0 * Math.PI * (d * d));
          dcGains[led.id] = coeff * Math.pow(cosPhi, m) * currentState.receiver.gain * cosPsi;
        } else {
          dcGains[led.id] = 0.0;
        }
      });

      const nextPath = [...currentState.trajectoryPoints, nextRxPos].slice(-120);
      setState((prev) => ({
        ...prev,
        currentTime: prev.currentTime + dt,
        frameIndex: prev.frameIndex + 1,
        receiver: { ...prev.receiver, position: nextRxPos, orientation: rxNormal, velocity: [vx, vy, vz] },
        distances, irradianceAngles, incidentAngles, losMatrix, visibilityMatrix, blockingObstacles, dcGains,
        trajectoryPoints: nextPath
      }));
    }, 50);
    return () => clearInterval(timer);
  }, [state.isPlaying, state.speedFactor]);

  // ─── Physics Engine Polling (every 10 frames → ~0.5s) ────────────────────
  const physicsFrameRef = useRef(0);
  const physicsInFlightRef = useRef(false);
  useEffect(() => {
    if (!state.isPlaying && state.physicsMetrics) return;
    if (physicsInFlightRef.current) return; // skip if previous call still running

    const POLL_EVERY = 10;
    physicsFrameRef.current = (physicsFrameRef.current + 1) % POLL_EVERY;
    if (physicsFrameRef.current !== 0) return;
    if (Object.keys(state.distances).length === 0) return;

    const payload = {
      current_time: state.currentTime,
      frame_index: state.frameIndex,
      fps: state.fps,
      receiver_position: state.receiver.position,
      receiver_orientation: state.receiver.orientation,
      receiver_velocity: state.receiver.velocity,
      receiver_acceleration: state.receiver.acceleration,
      receiver_angles: {
        roll: state.receiver.roll,
        pitch: state.receiver.pitch,
        yaw: state.receiver.yaw,
      },
      led_positions: Object.fromEntries(state.leds.map((l) => [l.id, l.position])),
      led_powers: Object.fromEntries(state.leds.map((l) => [l.id, l.power])),
      led_active: Object.fromEntries(state.leds.map((l) => [l.id, true])),
      led_orientations: Object.fromEntries(state.leds.map((l) => [l.id, l.orientation])),
      led_beam_angles: Object.fromEntries(state.leds.map((l) => [l.id, l.beamAngle])),
      room_dims: [state.room.width, state.room.length, state.room.height],
      distances: state.distances,
      incident_angles: state.incidentAngles,
      irradiance_angles: state.irradianceAngles,
      visibility_matrix: state.visibilityMatrix,
      los_matrix: state.losMatrix,
      blocking_obstacles: state.blockingObstacles,
      obstacles: state.obstacles,
    };

    physicsInFlightRef.current = true;
    setState((prev) => ({ ...prev, physicsLoading: true }));

    fetch("/api/physics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json())
      .then((data) => {
        physicsInFlightRef.current = false;
        if (data.success && data.physics) {
          setState((prev) => ({
            ...prev,
            physicsMetrics: data.physics,
            physicsLoading: false,
          }));
        } else {
          setState((prev) => ({ ...prev, physicsLoading: false }));
        }
      })
      .catch(() => {
        physicsInFlightRef.current = false;
        setState((prev) => ({ ...prev, physicsLoading: false }));
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.frameIndex]);

  // ─── Communication Engine Polling (every 30 frames → ~1.5s) ──────────────
  const commFrameRef = useRef(0);
  const commInFlightRef = useRef(false);
  useEffect(() => {
    if (!state.isPlaying && state.commMetrics) return;
    if (commInFlightRef.current) return; // skip if previous call still running

    const POLL_EVERY = 30;
    commFrameRef.current = (commFrameRef.current + 1) % POLL_EVERY;
    if (commFrameRef.current !== 0) return;
    if (Object.keys(state.distances).length === 0) return;

    const payload = {
      current_time: state.currentTime,
      frame_index: state.frameIndex,
      fps: state.fps,
      receiver_position: state.receiver.position,
      receiver_orientation: state.receiver.orientation,
      receiver_velocity: state.receiver.velocity,
      receiver_acceleration: state.receiver.acceleration,
      receiver_angles: {
        roll: state.receiver.roll,
        pitch: state.receiver.pitch,
        yaw: state.receiver.yaw,
      },
      led_positions: Object.fromEntries(state.leds.map((l) => [l.id, l.position])),
      led_powers: Object.fromEntries(state.leds.map((l) => [l.id, l.power])),
      led_active: Object.fromEntries(state.leds.map((l) => [l.id, true])),
      led_orientations: Object.fromEntries(state.leds.map((l) => [l.id, l.orientation])),
      led_beam_angles: Object.fromEntries(state.leds.map((l) => [l.id, l.beamAngle])),
      room_dims: [state.room.width, state.room.length, state.room.height],
      distances: state.distances,
      incident_angles: state.incidentAngles,
      irradiance_angles: state.irradianceAngles,
      visibility_matrix: state.visibilityMatrix,
      los_matrix: state.losMatrix,
      blocking_obstacles: state.blockingObstacles,
      obstacles: state.obstacles,
    };

    commInFlightRef.current = true;
    setState((prev) => ({ ...prev, commLoading: true }));

    fetch("/api/communication", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json())
      .then((data) => {
        commInFlightRef.current = false;
        if (data.success && data.communication) {
          setState((prev) => ({
            ...prev,
            commMetrics: data.communication,
            commLoading: false,
          }));
        } else {
          setState((prev) => ({ ...prev, commLoading: false }));
        }
      })
      .catch(() => {
        commInFlightRef.current = false;
        setState((prev) => ({ ...prev, commLoading: false }));
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.frameIndex]);

  // ─── Localization Engine Polling (every 60 frames → ~3s) ──────────────────
  const locFrameRef = useRef(0);
  const locInFlightRef = useRef(false);
  useEffect(() => {
    if (!state.isPlaying && state.localizationMetrics) return;
    if (locInFlightRef.current) return; // skip if previous call still running

    const POLL_EVERY = 60; // longer interval — localization is compute-heavy
    locFrameRef.current = (locFrameRef.current + 1) % POLL_EVERY;
    if (locFrameRef.current !== 0) return;
    if (Object.keys(state.distances).length === 0) return;

    const payload = {
      current_time: state.currentTime,
      frame_index: state.frameIndex,
      fps: state.fps,
      receiver_position: state.receiver.position,
      receiver_orientation: state.receiver.orientation,
      receiver_velocity: state.receiver.velocity,
      receiver_acceleration: state.receiver.acceleration,
      receiver_angles: {
        roll: state.receiver.roll,
        pitch: state.receiver.pitch,
        yaw: state.receiver.yaw,
      },
      led_positions: Object.fromEntries(state.leds.map((l) => [l.id, l.position])),
      led_powers: Object.fromEntries(state.leds.map((l) => [l.id, l.power])),
      led_active: Object.fromEntries(state.leds.map((l) => [l.id, true])),
      led_orientations: Object.fromEntries(state.leds.map((l) => [l.id, l.orientation])),
      led_beam_angles: Object.fromEntries(state.leds.map((l) => [l.id, l.beamAngle])),
      room_dims: [state.room.width, state.room.length, state.room.height],
      distances: state.distances,
      incident_angles: state.incidentAngles,
      irradiance_angles: state.irradianceAngles,
      visibility_matrix: state.visibilityMatrix,
      los_matrix: state.losMatrix,
      blocking_obstacles: state.blockingObstacles,
      obstacles: state.obstacles,
    };

    locInFlightRef.current = true;
    setState((prev) => ({ ...prev, localizationLoading: true }));

    fetch("/api/localization", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json())
      .then((data) => {
        locInFlightRef.current = false;
        if (data.success && data.localization) {
          setState((prev) => ({
            ...prev,
            localizationMetrics: data.localization,
            localizationLoading: false,
          }));
        } else {
          setState((prev) => ({ ...prev, localizationLoading: false }));
        }
      })
      .catch(() => {
        locInFlightRef.current = false;
        setState((prev) => ({ ...prev, localizationLoading: false }));
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.frameIndex]);

  // ─── Integrated Engine Polling (every 60 frames → ~3s) ────────────────────
  const integratedFrameRef = useRef(0);
  const integratedInFlightRef = useRef(false);
  useEffect(() => {
    if (!state.isPlaying && state.integratedMetrics) return;
    if (integratedInFlightRef.current) return;

    const POLL_EVERY = 60;
    integratedFrameRef.current = (integratedFrameRef.current + 1) % POLL_EVERY;
    if (integratedFrameRef.current !== 0) return;
    if (Object.keys(state.distances).length === 0) return;

    const payload = {
      current_time: state.currentTime,
      frame_index: state.frameIndex,
      fps: state.fps,
      receiver_position: state.receiver.position,
      receiver_orientation: state.receiver.orientation,
      receiver_velocity: state.receiver.velocity,
      receiver_acceleration: state.receiver.acceleration,
      receiver_angles: {
        roll: state.receiver.roll,
        pitch: state.receiver.pitch,
        yaw: state.receiver.yaw,
      },
      led_positions: Object.fromEntries(state.leds.map((l) => [l.id, l.position])),
      led_powers: Object.fromEntries(state.leds.map((l) => [l.id, l.power])),
      led_active: Object.fromEntries(state.leds.map((l) => [l.id, true])),
      led_orientations: Object.fromEntries(state.leds.map((l) => [l.id, l.orientation])),
      led_beam_angles: Object.fromEntries(state.leds.map((l) => [l.id, l.beamAngle])),
      room_dims: [state.room.width, state.room.length, state.room.height],
      distances: state.distances,
      incident_angles: state.incidentAngles,
      irradiance_angles: state.irradianceAngles,
      visibility_matrix: state.visibilityMatrix,
      los_matrix: state.losMatrix,
      blocking_obstacles: state.blockingObstacles,
      obstacles: state.obstacles,
    };

    integratedInFlightRef.current = true;
    setState((prev) => ({ ...prev, integratedLoading: true }));

    fetch("/api/integrated", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json())
      .then((data) => {
        integratedInFlightRef.current = false;
        if (data.success && data.integrated) {
          setState((prev) => ({
            ...prev,
            integratedMetrics: data.integrated,
            integratedLoading: false,
          }));
        } else {
          setState((prev) => ({ ...prev, integratedLoading: false }));
        }
      })
      .catch(() => {
        integratedInFlightRef.current = false;
        setState((prev) => ({ ...prev, integratedLoading: false }));
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.frameIndex]);

  // ─── Adaptive Engine Polling (every 60 frames → ~3s) ─────────────────────────────
  const adaptiveFrameRef = useRef(0);
  const adaptiveInFlightRef = useRef(false);
  useEffect(() => {
    if (!state.isPlaying && state.adaptiveMetrics) return;
    if (adaptiveInFlightRef.current) return;

    const POLL_EVERY = 60;
    adaptiveFrameRef.current = (adaptiveFrameRef.current + 1) % POLL_EVERY;
    if (adaptiveFrameRef.current !== 0) return;
    if (Object.keys(state.distances).length === 0) return;

    const payload = {
      current_time: state.currentTime,
      frame_index: state.frameIndex,
      fps: state.fps,
      receiver_position: state.receiver.position,
      receiver_orientation: state.receiver.orientation,
      receiver_velocity: state.receiver.velocity,
      receiver_acceleration: state.receiver.acceleration,
      receiver_angles: {
        roll: state.receiver.roll,
        pitch: state.receiver.pitch,
        yaw: state.receiver.yaw,
      },
      led_positions: Object.fromEntries(state.leds.map((l) => [l.id, l.position])),
      led_powers: Object.fromEntries(state.leds.map((l) => [l.id, l.power])),
      led_active: Object.fromEntries(state.leds.map((l) => [l.id, true])),
      led_orientations: Object.fromEntries(state.leds.map((l) => [l.id, l.orientation])),
      led_beam_angles: Object.fromEntries(state.leds.map((l) => [l.id, l.beamAngle])),
      room_dims: [state.room.width, state.room.length, state.room.height],
      distances: state.distances,
      incident_angles: state.incidentAngles,
      irradiance_angles: state.irradianceAngles,
      visibility_matrix: state.visibilityMatrix,
      los_matrix: state.losMatrix,
      blocking_obstacles: state.blockingObstacles,
      obstacles: state.obstacles,
      // Module 6 params
      num_devices: state.leds.length,
      min_rate_bps: 5e6,
      ber_max: 3.8e-3,
      mode: "ADAPTIVE",
      fft_size: 256,
    };

    adaptiveInFlightRef.current = true;
    setState((prev) => ({ ...prev, adaptiveLoading: true }));

    fetch("/api/adaptive", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json())
      .then((data) => {
        adaptiveInFlightRef.current = false;
        if (data.success && data.adaptive) {
          setState((prev) => ({
            ...prev,
            adaptiveMetrics: data.adaptive,
            adaptiveLoading: false,
          }));
        } else {
          setState((prev) => ({ ...prev, adaptiveLoading: false }));
        }
      })
      .catch(() => {
        adaptiveInFlightRef.current = false;
        setState((prev) => ({ ...prev, adaptiveLoading: false }));
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.frameIndex]);

  // ─── Power & Pre-EQ Engine Polling (every 90 frames → ~4.5s) ─────────────────
  const powerFrameRef = useRef(0);
  const powerInFlightRef = useRef(false);
  useEffect(() => {
    if (!state.isPlaying && state.powerMetrics) return;
    if (powerInFlightRef.current) return;

    const POLL_EVERY = 90;
    powerFrameRef.current = (powerFrameRef.current + 1) % POLL_EVERY;
    if (powerFrameRef.current !== 0) return;
    if (Object.keys(state.distances).length === 0) return;

    const payload = {
      current_time: state.currentTime,
      frame_index: state.frameIndex,
      fps: state.fps,
      receiver_position: state.receiver.position,
      receiver_orientation: state.receiver.orientation,
      receiver_velocity: state.receiver.velocity,
      receiver_acceleration: state.receiver.acceleration,
      receiver_angles: {
        roll: state.receiver.roll,
        pitch: state.receiver.pitch,
        yaw: state.receiver.yaw,
      },
      led_positions: Object.fromEntries(state.leds.map((l) => [l.id, l.position])),
      led_powers: Object.fromEntries(state.leds.map((l) => [l.id, l.power])),
      led_active: Object.fromEntries(state.leds.map((l) => [l.id, true])),
      led_orientations: Object.fromEntries(state.leds.map((l) => [l.id, l.orientation])),
      led_beam_angles: Object.fromEntries(state.leds.map((l) => [l.id, l.beamAngle])),
      room_dims: [state.room.width, state.room.length, state.room.height],
      distances: state.distances,
      incident_angles: state.incidentAngles,
      irradiance_angles: state.irradianceAngles,
      visibility_matrix: state.visibilityMatrix,
      los_matrix: state.losMatrix,
      blocking_obstacles: state.blockingObstacles,
      obstacles: state.obstacles,
      // Module 6 params (needed to chain into Module 7)
      num_devices: state.leds.length,
      min_rate_bps: 5e6,
      ber_max: 3.8e-3,
      mode: "ADAPTIVE",
      fft_size: 256,
      // Module 7 params
      total_power_budget_w: state.leds.reduce((s, l) => s + l.power, 0) * 0.001,
      power_mode: "EQUAL_POWER",
      pre_eq_mode: "REGULARIZED",
      localization_reserve_w: 0.1,
    };

    powerInFlightRef.current = true;
    setState((prev) => ({ ...prev, powerLoading: true }));

    fetch("/api/power", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json())
      .then((data) => {
        powerInFlightRef.current = false;
        if (data.success && data.power) {
          setState((prev) => ({
            ...prev,
            powerMetrics: data.power,
            powerLoading: false,
          }));
        } else {
          setState((prev) => ({ ...prev, powerLoading: false }));
        }
      })
      .catch(() => {
        powerInFlightRef.current = false;
        setState((prev) => ({ ...prev, powerLoading: false }));
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.frameIndex]);

  // ─── Joint Optimization Engine Polling (every 120 frames → ~6s) ────────────────
  const jointFrameRef = useRef(0);
  const jointInFlightRef = useRef(false);
  useEffect(() => {
    if (!state.isPlaying) return;
    if (jointInFlightRef.current) return;

    const POLL_EVERY = 120;
    jointFrameRef.current = (jointFrameRef.current + 1) % POLL_EVERY;
    if (jointFrameRef.current !== 0) return;
    if (Object.keys(state.distances).length === 0) return;

    const payload = {
      current_time: state.currentTime,
      frame_index: state.frameIndex,
      fps: state.fps,
      receiver_position: state.receiver.position,
      receiver_orientation: state.receiver.orientation,
      receiver_velocity: state.receiver.velocity,
      receiver_acceleration: state.receiver.acceleration,
      receiver_angles: {
        roll: state.receiver.roll,
        pitch: state.receiver.pitch,
        yaw: state.receiver.yaw,
      },
      led_positions: Object.fromEntries(state.leds.map((l) => [l.id, l.position])),
      led_powers: Object.fromEntries(state.leds.map((l) => [l.id, l.power])),
      led_active: Object.fromEntries(state.leds.map((l) => [l.id, true])),
      led_orientations: Object.fromEntries(state.leds.map((l) => [l.id, l.orientation])),
      led_beam_angles: Object.fromEntries(state.leds.map((l) => [l.id, l.beamAngle])),
      room_dims: [state.room.width, state.room.length, state.room.height],
      distances: state.distances,
      incident_angles: state.incidentAngles,
      irradiance_angles: state.irradianceAngles,
      visibility_matrix: state.visibilityMatrix,
      los_matrix: state.losMatrix,
      blocking_obstacles: state.blockingObstacles,
      obstacles: state.obstacles,
      // Joint Module 8 Params
      num_devices: state.leds.length,
      min_rate_bps: 5e6,
      ber_max: 3.8e-3,
      target_localization_error_m: 0.20,
      total_power_budget_w: 40.0,
      per_led_max_power_w: 10.0,
      power_mode: "WATER_FILLING",
      pre_eq_mode: "REGULARIZED",
      max_iterations: 6,
    };

    jointInFlightRef.current = true;
    setState((prev) => ({ ...prev, jointLoading: true }));

    fetch("/api/joint", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json())
      .then((data) => {
        jointInFlightRef.current = false;
        if (data.success && data.joint) {
          setState((prev) => ({
            ...prev,
            jointMetrics: data.joint,
            jointLoading: false,
          }));
        } else {
          setState((prev) => ({ ...prev, jointLoading: false }));
        }
      })
      .catch(() => {
        jointInFlightRef.current = false;
        setState((prev) => ({ ...prev, jointLoading: false }));
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.frameIndex]);

  // ─── 4. Handlers ──────────────────────────────────────────────────────────
  const handleDownloadZip = () => window.open("/api/export-zip");

  const handleReset = () => {
    timeElapsedRef.current = 0.0;
    waypointIndexRef.current = 0;
    randomWalkTimerRef.current = 0.0;
    setState((prev) => ({
      ...prev,
      currentTime: 0.0,
      frameIndex: 0,
      receiver: { ...prev.receiver, position: [2.5, 2.5, 0.85], orientation: [0, 0, 1], velocity: [0.2, 0.1, 0.0] },
      trajectoryPoints: [[2.5, 2.5, 0.85]]
    }));
    setPythonTerminalLogs("Simulation states re-initialized to initial coordinate parameters.");
    setPythonSuccess(null);
  };

  const generateYamlContent = () => `room:
  width: ${state.room.width.toFixed(1)}
  length: ${state.room.length.toFixed(1)}
  height: ${state.room.height.toFixed(1)}
  wall_reflectivity: ${state.room.wallReflectivity.toFixed(2)}
  floor_reflectivity: ${state.room.floorReflectivity.toFixed(2)}
  ceiling_reflectivity: ${state.room.ceilingReflectivity.toFixed(2)}

leds:
${state.leds.map((led) => `  - id: ${led.id}
    position: [${led.position.join(", ")}]
    orientation: [${led.orientation.join(", ")}]
    power: ${led.power.toFixed(1)}
    bias_current: ${led.biasCurrent.toFixed(1)}
    frequency: ${led.frequency.toFixed(1)}
    lambertian_order: ${led.lambertianOrder.toFixed(1)}
    beam_angle: ${led.beamAngle.toFixed(1)}
    fov: ${led.fov.toFixed(1)}
    communication_enabled: ${led.communicationEnabled}
    localization_enabled: ${led.localizationEnabled}`).join("\n")}

receiver:
  position: [${state.receiver.position.map((n) => n.toFixed(2)).join(", ")}]
  orientation: [${state.receiver.orientation.join(", ")}]
  velocity: [${state.receiver.velocity.join(", ")}]
  acceleration: [${state.receiver.acceleration.join(", ")}]
  fov: ${state.receiver.fov.toFixed(1)}
  apd_size: ${state.receiver.apdSize.toExponential(1)}
  noise: ${state.receiver.noise.toExponential(1)}
  gain: ${state.receiver.gain.toFixed(1)}
  roll: ${state.receiver.roll.toFixed(1)}
  pitch: ${state.receiver.pitch.toFixed(1)}
  yaw: ${state.receiver.yaw.toFixed(1)}

mobility:
  type: "${state.mobility.type}"
  speed: ${state.mobility.speed.toFixed(2)}
  radius: ${state.mobility.radius.toFixed(2)}
  center: [${state.mobility.center.join(", ")}]

obstacles:
${state.obstacles.map((obs) => `  - id: "${obs.id}"
    type: "${obs.type}"
    position: [${obs.position.join(", ")}]
    rotation: [${obs.rotation.join(", ")}]
    scale: [${obs.scale.join(", ")}]
    reflectivity: ${obs.reflectivity.toFixed(1)}
    material: "${obs.material}"`).join("\n")}
`;

  const handleExportYaml = () => {
    const yamlContent = generateYamlContent();
    const blob = new Blob([yamlContent], { type: "text/yaml;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "default.yaml");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleRunPythonSimulation = () => {
    setPythonLoading(true);
    setPythonTerminalLogs("Synchronizing configurations... Booting server side simulation sub-process...\n");
    const yamlContent = generateYamlContent();
    fetch("/api/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ content: yamlContent }) })
      .then(r => r.json())
      .then(d => {
        if (!d.success) throw new Error("Config sync failed.");
        setPythonTerminalLogs(p => p + "✓ Config synced.\n✓ Launching: python3 VLCL_AI/examples/demo_environment.py\n\n");
        return fetch("/api/run", { method: "POST" });
      })
      .then(r => r.json())
      .then(d => {
        setPythonLoading(false);
        if (d.success) {
          setPythonSuccess(true);
          setPythonTerminalLogs(p => p + d.stdout + "\n");
        } else {
          setPythonSuccess(false);
          setPythonTerminalLogs(p => p +
            `[ENGINE ERROR] Sub-process terminated.\nCause: Server container lacks full python runtime (numpy/loguru).\n\n` +
            `Output:\n${d.stderr || d.error}\n\n` +
            `========================================================================\n` +
            `  HOW TO RUN LOCALLY:\n` +
            `========================================================================\n` +
            `1. Click 'ZIP Package' in the left panel\n` +
            `2. Extract on your computer\n` +
            `3. pip install -r VLCL_AI/requirements.txt\n` +
            `4. python3 VLCL_AI/main.py\n` +
            `========================================================================\n`
          );
        }
        setActiveTab("terminal");
      })
      .catch(err => {
        setPythonLoading(false);
        setPythonSuccess(false);
        setPythonTerminalLogs(p => p + `Connection failed: ${err.message}\n`);
        setActiveTab("terminal");
      });
  };

  // ─── 5. Render ────────────────────────────────────────────────────────────
  return (
    <div className="h-screen w-screen bg-[#060c18] text-slate-300 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200 overflow-hidden">

      {/* ── Header ── */}
      <header className="border-b border-slate-800/80 bg-[#080f1e]/95 backdrop-blur-md shadow-xl shadow-black/20 flex-shrink-0 z-50">
        <div className="w-full px-6 py-2.5 flex items-center justify-between gap-4">

          {/* Brand */}
          <div className="flex items-center gap-4">
            <div className="relative w-8 h-8 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Activity className="w-4 h-4 text-white" />
              <span className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-emerald-500 rounded-full border-2 border-[#080f1e] animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-black text-white tracking-tight">VLCL_AI Dashboard</h1>
                <span className="text-cyan-400 font-mono text-[9px] bg-cyan-950/60 border border-cyan-900/40 px-1.5 py-0.5 rounded-md">v1.0.0-rc1</span>
              </div>
              <p className="text-[9px] text-slate-500 uppercase tracking-widest font-semibold">
                Visible Light Communication · Indoor Localization Simulator
              </p>
            </div>
          </div>

          {/* Tabs */}
          <nav className="flex bg-slate-950/80 p-0.5 rounded-xl border border-slate-800 shadow-inner gap-0.5 flex-shrink-0">
            {TABS.map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                title={tab.desc}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-bold transition-all duration-150 ${activeTab === tab.key
                    ? "bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                  }`}
              >
                {tab.icon}
                <span className="hidden sm:inline">{tab.label}</span>
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* ── Main Area (Fills remaining height) ── */}
      <main className="flex-1 min-h-0 w-full max-w-[1920px] mx-auto flex flex-col gap-3 p-3 overflow-hidden">

        {/* ── Dashboard Tab ── */}
        {activeTab === "visualizer" && (
          <div className="flex-1 min-h-0 w-full flex flex-row gap-3 overflow-hidden">

            {/* Left Column: Controls (25%) */}
            <div className="w-1/4 h-full flex flex-col min-w-[300px]">
              <ControlPanel
                state={state}
                setState={setState}
                onReset={handleReset}
              />
            </div>

            {/* Center Column: 3D Canvas (50%) */}
            <div className="w-2/4 h-full relative rounded-2xl overflow-hidden border border-slate-800 shadow-2xl bg-black">
              <ThreeCanvas state={state} localizationMetrics={state.localizationMetrics} />
              <ViewLegend />
            </div>

            {/* Right Column: Telemetry & Actions (25%) */}
            <div className="w-1/4 h-full min-h-0 flex flex-col gap-3 min-w-[300px]">

              {/* Telemetry Scrollable Panel */}
              <div className="flex-1 min-h-0 overflow-hidden border border-slate-800 rounded-2xl bg-[#0b1120] p-4 shadow-2xl">
                <DebugOverlay
                  state={state}
                  physicsMetrics={state.physicsMetrics}
                  physicsLoading={state.physicsLoading}
                  commMetrics={state.commMetrics}
                  commLoading={state.commLoading}
                  localizationMetrics={state.localizationMetrics}
                  localizationLoading={state.localizationLoading}
                  integratedMetrics={state.integratedMetrics}
                  integratedLoading={state.integratedLoading}
                  adaptiveMetrics={state.adaptiveMetrics}
                  adaptiveLoading={state.adaptiveLoading}
                  powerMetrics={state.powerMetrics}
                  powerLoading={state.powerLoading}
                  jointMetrics={state.jointMetrics}
                  jointLoading={state.jointLoading}
                />
              </div>

              {/* Action Buttons (Fixed at bottom right) */}
              <div className="bg-[#0b1120] border border-slate-800 rounded-2xl p-4 shadow-2xl flex-shrink-0">
                <div className="flex items-center gap-2 mb-3">
                  <Terminal className="w-4 h-4 text-cyan-400" />
                  <h4 className="font-bold text-xs text-slate-200">Export & Run Engine</h4>
                </div>
                <div className="flex flex-col gap-2">
                  <div className="flex gap-2">
                    <button
                      onClick={handleExportYaml}
                      className="flex-1 flex items-center justify-center gap-1.5 py-2 px-2 rounded-xl text-[10px] font-bold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
                    >
                      <Download className="w-3 h-3 text-cyan-400" />
                      Export YAML
                    </button>
                    <button
                      onClick={handleDownloadZip}
                      className="flex-1 flex items-center justify-center gap-1.5 py-2 px-2 rounded-xl text-[10px] font-bold bg-slate-900 hover:bg-slate-800 text-cyan-400 border border-cyan-500/30 hover:border-cyan-500/50 transition"
                    >
                      <Download className="w-3 h-3" />
                      ZIP Package
                    </button>
                  </div>
                  <button
                    onClick={handleRunPythonSimulation}
                    disabled={pythonLoading}
                    className={`w-full flex items-center justify-center gap-2 py-2.5 px-3 rounded-xl text-[11px] font-bold transition border ${pythonLoading
                        ? "bg-slate-800 text-slate-500 border-slate-700 cursor-not-allowed"
                        : pythonSuccess === true
                          ? "bg-emerald-900/40 text-emerald-300 border-emerald-800 hover:bg-emerald-900/60"
                          : pythonSuccess === false
                            ? "bg-rose-950/40 text-rose-300 border-rose-900 hover:bg-rose-950/60"
                            : "bg-cyan-500 hover:bg-cyan-400 text-slate-950 border-cyan-400 shadow-lg shadow-cyan-500/20"
                      }`}
                  >
                    <Terminal className={`w-3.5 h-3.5 ${pythonLoading || (pythonSuccess === null) ? "text-slate-950" : ""}`} />
                    {pythonLoading
                      ? "⏳ Running Engine..."
                      : pythonSuccess === true
                        ? "✅ Success — Run Again"
                        : pythonSuccess === false
                          ? "❌ Error — Retry"
                          : "▶  Run Python Engine"}
                  </button>
                  {pythonSuccess === true && (
                    <button
                      onClick={() => window.open("/api/visualization", "_blank")}
                      className="w-full flex items-center justify-center gap-2 py-2.5 px-3 rounded-xl text-[11px] font-bold transition border bg-cyan-900/40 text-cyan-300 border-cyan-800 hover:bg-cyan-900/60 shadow-lg"
                    >
                      <Layers className="w-3.5 h-3.5" />
                      View 3D HTML Visualization
                    </button>
                  )}
                </div>
              </div>
            </div>

          </div>
        )}

        {/* ── Live Results Tab ── */}
        {activeTab === "results" && <LiveResults state={state} telemetry={liveResults} />}

        {/* ── Dataset Factory Tab ── */}
        {activeTab === "dataset" && (
          <div className="flex-1 overflow-hidden flex flex-col">
            <DatasetPanel />
          </div>
        )}

        {activeTab === "guide" && (
          <div className="flex-1 overflow-y-auto custom-scrollbar">
            <IllustrationPanel />
          </div>
        )}

        {/* ── Formulas Tab ── */}
        {activeTab === "formulas" && (
          <div className="flex-1 overflow-y-auto custom-scrollbar">
            <FormulaPanel />
          </div>
        )}

        {/* ── Code Tab ── */}
        {activeTab === "code" && (
          <div className="flex-1 overflow-hidden">
            <CodeViewer />
          </div>
        )}

        {/* ── Terminal Tab ── */}
        {activeTab === "terminal" && (
          <div className="flex-1 bg-[#080f1e] border border-slate-800 rounded-2xl shadow-2xl flex flex-col gap-0 overflow-hidden">
            {/* Terminal titlebar */}
            <div className="flex justify-between items-center border-b border-slate-800 px-4 py-3 bg-slate-900/60 flex-shrink-0">
              <div className="flex items-center gap-3">
                <div className="flex gap-1.5">
                  <span className="w-3 h-3 rounded-full bg-red-500/80" />
                  <span className="w-3 h-3 rounded-full bg-yellow-500/80" />
                  <span className="w-3 h-3 rounded-full bg-emerald-500/80" />
                </div>
                <span className="text-slate-400 text-xs font-bold tracking-wider uppercase">Python Sub-Process Console</span>
              </div>
              <button
                onClick={() => setPythonTerminalLogs("")}
                className="px-2.5 py-1 bg-slate-800 border border-slate-700 rounded-lg text-[10px] hover:bg-slate-700 transition text-slate-400"
              >
                Clear
              </button>
            </div>
            {/* Terminal body */}
            <div className="flex-1 overflow-y-auto custom-scrollbar bg-[#020609] p-5">
              <pre className="text-cyan-400 text-xs leading-relaxed font-mono whitespace-pre-wrap">
                {pythonTerminalLogs || "$ Run the Python simulation from the control panel to see output here...\n$ Use 'Run Python Simulation' button in the 3D Simulator tab."}
              </pre>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}
