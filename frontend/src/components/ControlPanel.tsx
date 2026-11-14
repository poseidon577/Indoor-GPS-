/**
 * ControlPanel.tsx
 *
 * Redesigned for readability — every control has a clear label,
 * description, and visual cue so newcomers know what they're adjusting.
 */

import React, { useState } from "react";
import { Play, Pause, RotateCcw, Sliders, RefreshCw, ChevronDown, ChevronUp, Zap, Cpu, Move3D, Info } from "lucide-react";
import { SimulationState } from "../types";
import { REUSE_COLOURS_CSS, reuseColours } from "../lightLayout";

// ─── Props ─────────────────────────────────────────────────────────────────
interface ControlPanelProps {
  state: SimulationState;
  setState: React.Dispatch<React.SetStateAction<SimulationState>>;
  onReset: () => void;
  onLayout?: (n: number, spacing: number) => void;
}

// ─── Helper: Section Header ────────────────────────────────────────────────
function SectionHeader({
  icon,
  title,
  subtitle,
  color = "text-cyan-400",
}: {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  color?: string;
}) {
  return (
    <div className="flex items-start gap-2.5 mb-3">
      <div className={`mt-0.5 flex-shrink-0 ${color}`}>{icon}</div>
      <div>
        <h3 className="font-bold text-sm text-slate-100 leading-tight">{title}</h3>
        <p className="text-slate-500 text-xs leading-relaxed mt-0.5">{subtitle}</p>
      </div>
    </div>
  );
}

// ─── Helper: Labeled Slider ────────────────────────────────────────────────
function LabeledSlider({
  label,
  description,
  value,
  unit,
  min,
  max,
  step,
  onChange,
  accentClass = "accent-cyan",
  color = "text-cyan-400",
}: {
  label: string;
  description?: string;
  value: number;
  unit: string;
  min: number;
  max: number;
  step: number;
  onChange: (val: number) => void;
  accentClass?: string;
  color?: string;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between items-baseline">
        <span className="text-slate-300 text-xs font-semibold">{label}</span>
        <span className={`font-mono font-bold text-xs ${color}`}>
          {typeof value === "number" && value % 1 !== 0 ? value.toFixed(2) : value} {unit}
        </span>
      </div>
      {description && <p className="text-slate-600 text-[11px] leading-tight">{description}</p>}
      <div className="flex items-center gap-2">
        <span className="text-slate-600 text-[10px] font-mono w-8 text-right flex-shrink-0">{min}</span>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className={`w-full h-[4px] rounded appearance-none cursor-pointer ${accentClass}`}
        />
        <span className="text-slate-600 text-[10px] font-mono w-8 flex-shrink-0">{max}</span>
      </div>
    </div>
  );
}

// ─── Helper: Info Tip ──────────────────────────────────────────────────────
function InfoTip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative inline-block">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 text-[10px] text-slate-500 hover:text-cyan-400 transition mt-0.5"
      >
        <Info className="w-3 h-3" />
        <span>What does this do?</span>
      </button>
      {open && (
        <div className="absolute left-0 top-5 z-50 bg-slate-900 border border-slate-700 rounded-xl p-3 shadow-2xl text-xs text-slate-300 w-64 leading-relaxed animate-fade-in">
          {text}
          <button onClick={() => setOpen(false)} className="block mt-2 text-[10px] text-slate-500 hover:text-slate-300">
            Close
          </button>
        </div>
      )}
    </div>
  );
}

// ─── Helper: Collapsible Section ──────────────────────────────────────────
function CollapsibleSection({
  defaultOpen = true,
  children,
}: {
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div>
      <button
        className="w-full flex justify-end text-[10px] text-slate-600 hover:text-slate-400 transition mb-1"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? (
          <span className="flex items-center gap-1"><ChevronUp className="w-3 h-3" /> collapse</span>
        ) : (
          <span className="flex items-center gap-1"><ChevronDown className="w-3 h-3" /> expand</span>
        )}
      </button>
      {open && <div className="flex flex-col gap-3">{children}</div>}
    </div>
  );
}

// ─── Mobility type info ────────────────────────────────────────────────────
const MOBILITY_INFO: Record<string, { label: string; icon: string; desc: string }> = {
  static:      { label: "Static",      icon: "⬛", desc: "Receiver stays in one spot. Good for testing signal at a fixed location." },
  linear:      { label: "Linear",      icon: "➡️", desc: "Moves in a straight line, bouncing off walls. Like a Roomba." },
  circular:    { label: "Circular",    icon: "🔄", desc: "Orbits around the room center. Great for seeing how signal changes with position." },
  random_walk: { label: "Random Walk", icon: "🎲", desc: "Moves in random directions, changing every 2 seconds. Simulates unpredictable movement." },
  waypoint:    { label: "Waypoint",    icon: "📍", desc: "Follows a preset route visiting 4 corner points in order." },
};

// ─── Main Component ────────────────────────────────────────────────────────
export default function ControlPanel({
  state,
  setState,
  onReset,
  onLayout,
}: ControlPanelProps) {
  const [gridSpacing, setGridSpacing] = useState(1.5);
  const gridN = state.leds.length > 4 ? Math.round(Math.sqrt(state.leds.length)) : 2;
  const colourOf = reuseColours(state.leds);
  const colourCounts = [1, 2, 3, 4].map((c) => Object.values(colourOf).filter((v) => v === c).length);

  const handleTogglePlay = () => setState((prev) => ({ ...prev, isPlaying: !prev.isPlaying }));

  const handleSpeedFactor = (val: number) => setState((prev) => ({ ...prev, speedFactor: val }));

  const handleTrajectoryChange = (type: any) =>
    setState((prev) => ({ ...prev, mobility: { ...prev.mobility, type }, trajectoryPoints: [] }));

  const updateLED = (id: number, field: string, val: any) =>
    setState((prev) => ({
      ...prev,
      leds: prev.leds.map((led) => (led.id === id ? { ...led, [field]: val } : led)),
    }));

  const updateReceiver = (field: string, val: any) =>
    setState((prev) => ({ ...prev, receiver: { ...prev.receiver, [field]: val } }));

  const updateObstacle = (id: string, field: string, index: number, val: number) =>
    setState((prev) => ({
      ...prev,
      obstacles: prev.obstacles.map((obs) => {
        if (obs.id === id && (field === "position" || field === "scale" || field === "rotation")) {
          const arr = [...obs[field]] as [number, number, number];
          arr[index] = val;
          return { ...obs, [field]: arr };
        }
        return obs;
      }),
    }));

  const selectedMobility = MOBILITY_INFO[state.mobility.type];

  return (
    <div className="w-full h-full flex flex-col gap-0 bg-[#0b1120] border border-slate-800 rounded-2xl shadow-2xl text-slate-300 overflow-hidden">

      {/* ── Panel Title ── */}
      <div className="bg-gradient-to-r from-slate-900 to-[#0b1120] border-b border-slate-800 px-4 pt-4 pb-3 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center">
            <Sliders className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <h2 className="font-bold text-sm text-white">Simulation Controls</h2>
            <p className="text-slate-500 text-[10px]">Adjust parameters in real-time</p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col divide-y divide-slate-800/60 pb-4">
        {/* ── Section 1: Playback ── */}
        <div className="px-4 py-4">
          <SectionHeader
            icon={<Zap className="w-4 h-4" />}
            title="Simulation Engine"
            subtitle="Start or pause physics. Speed controls time advance."
          />
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={handleTogglePlay}
              className={`flex flex-1 items-center justify-center gap-2 px-3 py-2 rounded-xl text-xs font-bold transition-all duration-200 shadow-md ${
                state.isPlaying
                  ? "bg-amber-500 hover:bg-amber-600 text-slate-950 shadow-amber-500/20"
                  : "bg-emerald-500 hover:bg-emerald-600 text-slate-950 shadow-emerald-500/20"
              }`}
            >
              {state.isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              {state.isPlaying ? "Pause" : "Start"}
            </button>

            <button
              onClick={onReset}
              className="flex items-center justify-center gap-1.5 px-3 py-2 rounded-xl text-xs font-bold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
            >
              <RotateCcw className="w-3.5 h-3.5 text-cyan-400" />
              Reset
            </button>

            {/* Speed selector */}
            <div className="flex bg-slate-950 rounded-xl p-0.5 border border-slate-800">
              {[["0.5", "½×"], ["1.0", "1×"], ["2.0", "2×"]].map(([val, label]) => (
                <button
                  key={val}
                  onClick={() => handleSpeedFactor(parseFloat(val))}
                  className={`px-2 py-1 text-[11px] font-bold rounded-lg transition ${
                    state.speedFactor === parseFloat(val)
                      ? "bg-cyan-500 text-slate-950 shadow-sm"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* ── Section 2: Mobility ── */}
        <div className="px-4 py-4">
          <SectionHeader
            icon={<RefreshCw className="w-4 h-4 animate-spin-slow" />}
            title="Receiver Movement Pattern"
            subtitle="Choose how the mobile receiver moves around."
          />

          <div className="grid grid-cols-2 gap-2">
            {Object.entries(MOBILITY_INFO).map(([type, info]) => (
              <button
                key={type}
                onClick={() => handleTrajectoryChange(type as any)}
                className={`flex flex-col gap-0.5 px-3 py-2 rounded-xl text-left transition-all border ${
                  state.mobility.type === type
                    ? "bg-cyan-500/15 border-cyan-500/50 shadow-inner"
                    : "bg-slate-900/50 hover:bg-slate-800/60 border-slate-800 hover:border-slate-700"
                }`}
              >
                <span className="text-sm leading-none">{info.icon}</span>
                <span className={`font-bold text-[10px] ${state.mobility.type === type ? "text-cyan-300" : "text-slate-400"}`}>
                  {info.label}
                </span>
              </button>
            ))}
          </div>

          {selectedMobility && (
            <div className="mt-2 flex items-start gap-1.5 bg-slate-900/50 border border-slate-800 rounded-xl px-2.5 py-1.5">
              <Info className="w-3 h-3 text-cyan-400 flex-shrink-0 mt-0.5" />
              <p className="text-slate-400 text-[10px] leading-relaxed">{selectedMobility.desc}</p>
            </div>
          )}

          {state.mobility.type !== "static" && (
            <div className="mt-3">
              <LabeledSlider
                label="Movement Speed"
                value={state.mobility.speed}
                unit="m/s"
                min={0.1}
                max={2.0}
                step={0.05}
                onChange={(val) => setState((prev) => ({ ...prev, mobility: { ...prev.mobility, speed: val } }))}
              />
            </div>
          )}
        </div>

        {/* ── Section 2b: Ceiling light layout ── */}
        {onLayout && (
          <div className="px-4 py-4">
            <SectionHeader
              icon={<Cpu className="w-4 h-4" />}
              title="Ceiling Light Layout"
              subtitle="Choose how many LEDs are on the ceiling. Bigger grids make a bigger room."
              color="text-amber-300"
            />
            <div className="grid grid-cols-3 gap-1.5">
              {[[2, "4 (paper)"], [4, "4×4 = 16"], [6, "6×6 = 36"]].map(([n, label]) => (
                <button
                  key={n as number}
                  data-testid={`layout-${n}`}
                  onClick={() => onLayout(n as number, gridSpacing)}
                  className={`px-1.5 py-1.5 text-[11px] font-bold rounded-lg border transition ${
                    gridN === n
                      ? "bg-amber-400 text-slate-950 border-amber-300"
                      : "bg-slate-900 text-slate-300 border-slate-700 hover:bg-slate-800"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="mt-3">
              <LabeledSlider
                label="Grid spacing (between neighbouring LEDs)"
                value={gridSpacing}
                unit=" m"
                min={0.8}
                max={2.5}
                step={0.1}
                onChange={(val) => {
                  setGridSpacing(val);
                  if (state.leds.length > 4) onLayout(gridN, val);
                }}
              />
            </div>
            <div className="mt-3 flex items-center gap-2 flex-wrap">
              {colourCounts.map((count, i) => (
                <span key={i} className="flex items-center gap-1 text-[10px] text-slate-400">
                  <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: REUSE_COLOURS_CSS[i] }} />
                  {i === 0 ? "tones 1+5" : `tone ${i + 1}`} × {count}
                </span>
              ))}
            </div>
            <p className="text-slate-500 text-[10px] leading-relaxed mt-2">
              There are only a few frequency tones, so lights with the same colour reuse the same tones.
              With more than 4 LEDs the Localization panel compares the interference-aware solver
              against the old "nearest light only" assumption. Odd grids such as 3×3 are left out on purpose:
              their colour pattern looks the same after a half turn, so a position and its mirror image give identical signals.
            </p>
          </div>
        )}

        {/* ── Section 3: LED Powers ── */}
        <div className="px-4 py-4">
          <SectionHeader
            icon={<Cpu className="w-4 h-4" />}
            title="LED Transmitter Power"
            subtitle="Higher power = stronger signal."
          />
          <div className="flex flex-col gap-3 mt-2">
            {state.leds.length > 4 && (
              <LabeledSlider
                label={`All ${state.leds.length} LEDs`}
                value={state.leds[0].power}
                unit="W"
                min={5}
                max={50}
                step={1}
                onChange={(val) => setState((prev) => ({ ...prev, leds: prev.leds.map((l) => ({ ...l, power: val })) }))}
              />
            )}
            {state.leds.length <= 4 && state.leds.map((led) => (
              <LabeledSlider
                key={led.id}
                label={`LED ${led.id} (${led.position[0]}, ${led.position[1]}, ${led.position[2]})`}
                value={led.power}
                unit="W"
                min={5}
                max={50}
                step={1}
                onChange={(val) => updateLED(led.id, "power", val)}
              />
            ))}
          </div>
        </div>

        {/* ── Section 4: Receiver Settings ── */}
        <div className="px-4 py-4">
          <SectionHeader
            icon={<Move3D className="w-4 h-4" />}
            title="Receiver Sensor Settings"
            subtitle="Configure view angle and tilt."
          />
          <CollapsibleSection>
            <div>
              <LabeledSlider
                label="Field of View (FOV)"
                value={state.receiver.fov}
                unit="°"
                min={30}
                max={90}
                step={2}
                onChange={(val) => updateReceiver("fov", val)}
              />
            </div>
            <div className="flex flex-col gap-2 mt-2">
              <p className="text-[10px] font-semibold text-slate-400">Sensor Tilt Angles</p>
              <div className="grid grid-cols-3 gap-2">
                {(
                  [
                    { key: "roll", label: "Roll", range: [-45, 45], color: "text-amber-400" },
                    { key: "pitch", label: "Pitch", range: [-45, 45], color: "text-violet-400" },
                    { key: "yaw", label: "Yaw", range: [-180, 180], color: "text-emerald-400" },
                  ] as const
                ).map(({ key, label, range, color }) => (
                  <div key={key} className="flex flex-col gap-1 bg-slate-900/60 border border-slate-800 rounded-xl p-1.5">
                    <span className={`text-[9px] font-bold uppercase tracking-wider text-center ${color}`}>{label}</span>
                    <input
                      type="number"
                      min={range[0]}
                      max={range[1]}
                      value={(state.receiver as any)[key]}
                      onChange={(e) => updateReceiver(key, parseFloat(e.target.value) || 0)}
                      className={`bg-slate-950 border border-slate-800 rounded-lg px-1 py-1 font-mono text-[10px] ${color} focus:outline-none focus:border-cyan-500 text-center w-full`}
                    />
                  </div>
                ))}
              </div>
            </div>
          </CollapsibleSection>
        </div>

        {/* ── Section 5: Obstacles ── */}
        <div className="px-4 py-4">
          <SectionHeader
            icon={<span className="text-base">🚧</span>}
            title="Obstacle Positions"
            subtitle="Drag human obstacle to block LED paths."
            color="text-rose-400"
          />
          <div className="flex flex-col gap-3 bg-rose-950/10 border border-rose-900/20 rounded-xl p-3 mt-2">
            <LabeledSlider
              label="Human — X Position"
              value={state.obstacles[0].position[0]}
              unit="m"
              min={0.5}
              max={4.5}
              step={0.05}
              onChange={(val) => updateObstacle("obs_human", "position", 0, val)}
              accentClass="accent-rose-500"
              color="text-rose-400"
            />
            <LabeledSlider
              label="Human — Y Position"
              value={state.obstacles[0].position[1]}
              unit="m"
              min={0.5}
              max={4.5}
              step={0.05}
              onChange={(val) => updateObstacle("obs_human", "position", 1, val)}
              accentClass="accent-rose-500"
              color="text-rose-400"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
