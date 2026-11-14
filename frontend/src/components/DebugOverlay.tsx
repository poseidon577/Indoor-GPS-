/**
 * DebugOverlay.tsx
 *
 * Real-time simulation statistics panel.
 * - Plain-English labels, color-coded status badges
 * - Visual signal strength progress bars (JS geometry)
 * - Physics Engine Metrics panel (SNR, optical power, currents — Python backend)
 */

import { Cpu, Wifi, ShieldAlert, Navigation, Clock, Signal, CheckCircle2, XCircle, EyeOff, Zap, Activity, AlertCircle, Loader } from "lucide-react";
import { SimulationState, PhysicsMetrics, CommunicationMetrics, LocalizationMetrics, MulticellMetrics, IntegratedMetrics, AdaptiveMetrics, PowerMetrics, JointMetrics } from "../types";
import CommunicationPanel from "./CommunicationPanel";
import LocalizationPanel from "./LocalizationPanel";
import MultiLightPanel from "./MultiLightPanel";
import IntegratedPanel from "./IntegratedPanel";
import AdaptivePanel from "./AdaptivePanel";
import PowerPanel from "./PowerPanel";
import { JointPanel } from "./JointPanel";

interface DebugOverlayProps {
  state: SimulationState;
  physicsMetrics: PhysicsMetrics | null;
  physicsLoading: boolean;
  commMetrics: CommunicationMetrics | null;
  commLoading: boolean;
  localizationMetrics: LocalizationMetrics | null;
  localizationLoading: boolean;
  multicellMetrics?: MulticellMetrics | null;
  multicellLoading?: boolean;
  integratedMetrics: IntegratedMetrics | null;
  integratedLoading: boolean;
  adaptiveMetrics: AdaptiveMetrics | null;
  adaptiveLoading: boolean;
  powerMetrics: PowerMetrics | null;
  powerLoading: boolean;
  jointMetrics: JointMetrics | null;
  jointLoading: boolean;
}

// ─── Stat Row ──────────────────────────────────────────────────────────────
function StatRow({
  label,
  value,
  valueClass = "text-slate-300",
  sub,
}: {
  label: string;
  value: string | React.ReactNode;
  valueClass?: string;
  sub?: string;
}) {
  return (
    <div className="flex justify-between items-start gap-2">
      <div className="flex flex-col">
        <span className="text-slate-500 text-xs">{label}</span>
        {sub && <span className="text-[10px] text-slate-700 italic">{sub}</span>}
      </div>
      <span className={`font-mono font-bold text-xs text-right ${valueClass}`}>{value}</span>
    </div>
  );
}

// ─── Signal Status Badge ───────────────────────────────────────────────────
function SignalBadge({ isLos, isFov }: { isLos: boolean; isFov: boolean }) {
  if (!isLos) {
    return (
      <span className="inline-flex items-center gap-1 bg-rose-950/80 text-rose-400 border border-rose-900/40 px-1.5 py-0.5 rounded-lg text-[9px] font-bold">
        <ShieldAlert className="w-2.5 h-2.5" /> BLOCKED
      </span>
    );
  }
  if (!isFov) {
    return (
      <span className="inline-flex items-center gap-1 bg-slate-800 text-slate-400 border border-slate-700 px-1.5 py-0.5 rounded-lg text-[9px] font-bold">
        <EyeOff className="w-2.5 h-2.5" /> OUT OF FOV
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 bg-emerald-950/80 text-emerald-400 border border-emerald-900/40 px-1.5 py-0.5 rounded-lg text-[9px] font-bold">
      <CheckCircle2 className="w-2.5 h-2.5" /> CLEAR LOS
    </span>
  );
}

// ─── LED Signal Card (JS geometry) ────────────────────────────────────────
function LEDSignalCard({ led, dist, gain, isLos, isFov }: {
  led: { id: number; power: number };
  dist: number;
  gain: number;
  isLos: boolean;
  isFov: boolean;
}) {
  const gainPercent = Math.min(100, Math.max(0, Math.round((gain / 1.2e-4) * 100)));
  const active = isLos && isFov;

  return (
    <div className={`flex flex-col gap-2 p-3 rounded-xl border transition-colors ${
      active
        ? "bg-emerald-950/20 border-emerald-900/40"
        : "bg-slate-900/40 border-slate-800"
    }`}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full flex-shrink-0 ${active ? "bg-emerald-400 shadow-sm shadow-emerald-400/50" : "bg-slate-600"}`} />
          <span className="font-bold text-xs text-slate-200">LED {led.id}</span>
          <span className="text-slate-600 text-[10px] font-mono">{led.power}W</span>
        </div>
        <SignalBadge isLos={isLos} isFov={isFov} />
      </div>

      <div className="flex justify-between text-[11px]">
        <span className="text-slate-500">
          Distance: <span className="text-slate-300 font-mono font-semibold">{dist.toFixed(2)} m</span>
        </span>
        <span className="text-slate-500">
          Gain H(0): <span className={`font-mono font-bold ${active ? "text-cyan-400" : "text-slate-600"}`}>
            {gain > 0 ? gain.toExponential(2) : "0"}
          </span>
        </span>
      </div>

      <div className="flex flex-col gap-1">
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-slate-600">Signal Strength</span>
          <span className={`text-[10px] font-mono font-bold ${active ? "text-emerald-400" : "text-slate-600"}`}>
            {active ? gainPercent : 0}%
          </span>
        </div>
        <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              active
                ? gainPercent > 60
                  ? "bg-gradient-to-r from-emerald-500 to-emerald-400"
                  : gainPercent > 25
                  ? "bg-gradient-to-r from-amber-500 to-amber-400"
                  : "bg-gradient-to-r from-rose-500 to-rose-400"
                : "bg-slate-700"
            }`}
            style={{ width: `${active ? gainPercent : 0}%` }}
          />
        </div>
      </div>
    </div>
  );
}

// ─── SNR Category Badge ────────────────────────────────────────────────────
function SnrBadge({ snr }: { snr: number }) {
  if (snr > 25)
    return <span className="px-1.5 py-0.5 text-[9px] font-bold rounded-md bg-emerald-950/80 text-emerald-400 border border-emerald-900/40">STRONG</span>;
  if (snr > 15)
    return <span className="px-1.5 py-0.5 text-[9px] font-bold rounded-md bg-amber-950/80 text-amber-400 border border-amber-900/40">MODERATE</span>;
  if (snr > 0)
    return <span className="px-1.5 py-0.5 text-[9px] font-bold rounded-md bg-rose-950/80 text-rose-400 border border-rose-900/40">WEAK</span>;
  return <span className="px-1.5 py-0.5 text-[9px] font-bold rounded-md bg-slate-800 text-slate-500 border border-slate-700">NO SIG</span>;
}

// ─── Physics LED Card ──────────────────────────────────────────────────────
function PhysicsLEDCard({ ledId, snr, power, current, losGain, nlosGain }: {
  ledId: string;
  snr: number;
  power: number;
  current: number;
  losGain: number;
  nlosGain: number;
}) {
  const snrPct = Math.min(100, Math.max(0, Math.round(((snr + 10) / 50) * 100)));
  const snrColor =
    snr > 25 ? "from-emerald-500 to-emerald-400" :
    snr > 15 ? "from-amber-500 to-amber-400" :
    snr > 0  ? "from-rose-500 to-rose-400"  :
               "from-slate-700 to-slate-700";

  return (
    <div className="flex flex-col gap-2 p-3 rounded-xl border bg-slate-950/60 border-slate-800">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <Zap className="w-3 h-3 text-violet-400" />
          <span className="text-xs font-bold text-slate-200">LED {ledId}</span>
        </div>
        <SnrBadge snr={snr} />
      </div>

      {/* SNR bar */}
      <div className="flex flex-col gap-1">
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-slate-500">SNR</span>
          <span className="text-[11px] font-mono font-bold text-violet-300">{snr.toFixed(1)} dB</span>
        </div>
        <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full bg-gradient-to-r ${snrColor} transition-all duration-700`}
            style={{ width: `${snrPct}%` }}
          />
        </div>
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px]">
        <div>
          <span className="text-slate-600">Rx Power</span>
          <div className="font-mono font-bold text-cyan-400">{(power * 1e6).toFixed(3)} µW</div>
        </div>
        <div>
          <span className="text-slate-600">PD Current</span>
          <div className="font-mono font-bold text-teal-400">{(current * 1e6).toFixed(3)} µA</div>
        </div>
        <div>
          <span className="text-slate-600">LOS Gain</span>
          <div className="font-mono font-bold text-slate-400">{losGain.toExponential(2)}</div>
        </div>
        <div>
          <span className="text-slate-600">NLOS Gain</span>
          <div className="font-mono font-bold text-slate-500">{nlosGain.toExponential(2)}</div>
        </div>
      </div>
    </div>
  );
}

// ─── Physics Engine Panel ──────────────────────────────────────────────────
function PhysicsPanel({ metrics, loading }: { metrics: PhysicsMetrics | null; loading: boolean }) {
  const ledIds = metrics ? Object.keys(metrics.snrs) : [];
  const coverageColor =
    metrics?.metrics.average_snr !== undefined
      ? metrics.metrics.average_snr > 25
        ? "text-emerald-400"
        : metrics.metrics.average_snr > 15
        ? "text-amber-400"
        : "text-rose-400"
      : "text-slate-500";

  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2">
          <Activity className="w-3.5 h-3.5 text-violet-400" />
          <h4 className="font-bold text-xs text-slate-200">Physics Engine Metrics</h4>
        </div>
        {loading && (
          <span className="flex items-center gap-1 text-[10px] text-violet-400">
            <Loader className="w-3 h-3 animate-spin" /> Computing...
          </span>
        )}
        {!loading && !metrics && (
          <span className="flex items-center gap-1 text-[10px] text-slate-600">
            <AlertCircle className="w-3 h-3" /> Python offline
          </span>
        )}
        {!loading && metrics && (
          <span className="flex items-center gap-1 text-[10px] text-emerald-400">
            <CheckCircle2 className="w-3 h-3" /> Live
          </span>
        )}
      </div>

      {!metrics && !loading && (
        <div className="text-center py-4">
          <AlertCircle className="w-6 h-6 text-slate-700 mx-auto mb-2" />
          <p className="text-[11px] text-slate-600 leading-relaxed">
            Physics engine metrics require the Python venv.<br />
            Run the simulation to activate.
          </p>
        </div>
      )}

      {metrics && (
        <>
          {/* Summary */}
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-slate-950/60 rounded-xl p-3">
              <span className="text-[10px] text-slate-500 block mb-0.5">Avg SNR</span>
              <span className={`font-mono font-black text-lg ${coverageColor}`}>
                {metrics.metrics.average_snr.toFixed(1)}
              </span>
              <span className="text-[10px] text-slate-600 ml-0.5">dB</span>
            </div>
            <div className="bg-slate-950/60 rounded-xl p-3">
              <span className="text-[10px] text-slate-500 block mb-0.5">Total Pwr</span>
              <span className="font-mono font-black text-lg text-cyan-400">
                {(metrics.metrics.total_optical_power * 1e6).toFixed(2)}
              </span>
              <span className="text-[10px] text-slate-600 ml-0.5">µW</span>
            </div>
            <div className="bg-slate-950/60 rounded-xl p-3">
              <span className="text-[10px] text-slate-500 block mb-0.5">Visible</span>
              <span className="font-mono font-black text-lg text-teal-400">
                {metrics.metrics.visible_leds}
              </span>
              <span className="text-[10px] text-slate-600 ml-0.5">LEDs</span>
            </div>
            <div className="bg-slate-950/60 rounded-xl p-3">
              <span className="text-[10px] text-slate-500 block mb-0.5">Blocked</span>
              <span className={`font-mono font-black text-lg ${metrics.metrics.blocked_leds > 0 ? "text-rose-400" : "text-emerald-400"}`}>
                {metrics.metrics.blocked_leds}
              </span>
              <span className="text-[10px] text-slate-600 ml-0.5">LEDs</span>
            </div>
          </div>

          {/* Per-LED physics cards */}
          <div className="flex flex-col gap-2">
            {ledIds.map((id) => (
              <PhysicsLEDCard
                key={id}
                ledId={id}
                snr={metrics.snrs[id] ?? 0}
                power={metrics.received_powers[id] ?? 0}
                current={metrics.electrical_currents[id] ?? 0}
                losGain={metrics.los_gains[id] ?? 0}
                nlosGain={metrics.nlos_gains[id] ?? 0}
              />
            ))}
          </div>

          <StatRow
            label="Propagation Delay"
            value={`${(metrics.metrics.propagation_delay * 1e9).toFixed(2)} ns`}
            valueClass="text-slate-400"
          />
          <StatRow
            label="Avg Channel Gain"
            value={metrics.metrics.average_channel_gain.toExponential(3)}
            valueClass="text-slate-400"
          />
        </>
      )}
    </div>
  );
}

// ─── Main Export ───────────────────────────────────────────────────────────
export default function DebugOverlay({ state, physicsMetrics, physicsLoading, commMetrics, commLoading, localizationMetrics, localizationLoading, multicellMetrics, multicellLoading, integratedMetrics, integratedLoading, adaptiveMetrics, adaptiveLoading,
  powerMetrics,
  powerLoading,
  jointMetrics,
  jointLoading
}: DebugOverlayProps) {
  const activeLeds = state.leds.filter(led => state.losMatrix[led.id] && state.visibilityMatrix[led.id]);

  return (
    <div className="w-full h-full flex flex-col gap-3 overflow-hidden">

      {/* Title bar */}
      <div className="flex items-center justify-between flex-shrink-0">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
          <Signal className="w-3.5 h-3.5 text-cyan-400" />
          Live Telemetry
        </h3>
        <span className={`flex items-center gap-1.5 text-[10px] font-bold px-2.5 py-1 rounded-full border ${
          state.isPlaying
            ? "bg-emerald-950/60 text-emerald-400 border-emerald-900/50"
            : "bg-slate-800 text-slate-500 border-slate-700"
        }`}>
          <span className={`w-1.5 h-1.5 rounded-full ${state.isPlaying ? "bg-emerald-400 animate-pulse" : "bg-slate-600"}`} />
          {state.isPlaying ? "LIVE" : "PAUSED"}
        </span>
      </div>

      {/* Stats vertical list */}
      <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar flex flex-col gap-3 pr-1 pb-4 [&>*]:flex-shrink-0">

        {/* ── Clock Panel ── */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-4 flex flex-col gap-3">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
            <Clock className="w-3.5 h-3.5 text-cyan-400" />
            <h4 className="font-bold text-xs text-slate-200">Simulation Clock</h4>
          </div>
          <div className="flex flex-col gap-2">
            <StatRow
              label="Elapsed Time"
              value={`${state.currentTime.toFixed(2)} s`}
              valueClass="text-cyan-400"
            />
            <StatRow
              label="Frame #"
              value={state.frameIndex.toString()}
              valueClass="text-slate-300"
            />
            <StatRow
              label="Render Rate"
              value={`${state.fps.toFixed(1)} FPS`}
              valueClass="text-emerald-400"
            />
            <StatRow
              label="Time Step (dt)"
              value="0.05 s"
              valueClass="text-slate-400"
              sub="per physics frame"
            />
          </div>
        </div>

        {/* ── Receiver Panel ── */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-4 flex flex-col gap-3">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
            <Navigation className="w-3.5 h-3.5 text-cyan-400" />
            <h4 className="font-bold text-xs text-slate-200">Receiver Position</h4>
          </div>
          <div className="flex flex-col gap-2">
            <StatRow
              label="Position X,Y,Z"
              value={`[${state.receiver.position.map((n) => n.toFixed(2)).join(", ")}]`}
              valueClass="text-cyan-400"
              sub="in meters"
            />
            <StatRow
              label="Sensor Normal"
              value={`[${state.receiver.orientation.map((n) => n.toFixed(2)).join(", ")}]`}
              valueClass="text-slate-300"
              sub="pointing direction"
            />
            <StatRow
              label="Tilt (R, P, Y)"
              value={`${state.receiver.roll.toFixed(1)}°, ${state.receiver.pitch.toFixed(1)}°, ${state.receiver.yaw.toFixed(1)}°`}
              valueClass="text-violet-300"
            />
            <StatRow
              label="Movement"
              value={state.mobility.type.replace("_", " ")}
              valueClass="text-cyan-400"
            />
          </div>
        </div>

        {/* ── Signal Overview ── */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-4 flex flex-col gap-3">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
            <Wifi className="w-3.5 h-3.5 text-cyan-400" />
            <h4 className="font-bold text-xs text-slate-200">Signal Summary</h4>
          </div>
          <div className="flex flex-col gap-2">
            <div className="flex flex-col gap-1 bg-slate-950/50 rounded-xl p-3">
              <span className="text-slate-500 text-[11px]">Active LEDs (LOS + in FOV)</span>
              <div className="flex items-baseline gap-1">
                <span className={`font-black text-2xl ${activeLeds.length > 0 ? "text-emerald-400" : "text-rose-400"}`}>
                  {activeLeds.length}
                </span>
                <span className="text-slate-500 text-sm">/ {state.leds.length}</span>
              </div>
              <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden mt-1">
                <div
                  className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 rounded-full transition-all duration-500"
                  style={{ width: `${(activeLeds.length / state.leds.length) * 100}%` }}
                />
              </div>
            </div>

            <div className="flex gap-2">
              {state.leds.map(led => {
                const isLos = state.losMatrix[led.id];
                const isFov = state.visibilityMatrix[led.id];
                const active = isLos && isFov;
                return (
                  <div key={led.id} className="flex flex-col items-center gap-1">
                    <div className={`w-5 h-5 rounded-full flex items-center justify-center border ${
                      active ? "bg-emerald-500/20 border-emerald-500/50" : !isLos ? "bg-rose-500/20 border-rose-500/50" : "bg-slate-800 border-slate-700"
                    }`}>
                      {active
                        ? <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                        : !isLos
                        ? <XCircle className="w-3 h-3 text-rose-400" />
                        : <EyeOff className="w-3 h-3 text-slate-500" />}
                    </div>
                    <span className="text-[9px] text-slate-600">L{led.id}</span>
                  </div>
                );
              })}
            </div>

            <div className="flex flex-col gap-0.5">
              <div className="flex items-center gap-1.5">
                <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                <span className="text-slate-500 text-[11px]">Clear line-of-sight, in view cone</span>
              </div>
              <div className="flex items-center gap-1.5">
                <XCircle className="w-3 h-3 text-rose-400" />
                <span className="text-slate-500 text-[11px]">Blocked by obstacle (NLOS)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <EyeOff className="w-3 h-3 text-slate-500" />
                <span className="text-slate-500 text-[11px]">Outside receiver FOV cone</span>
              </div>
            </div>
          </div>
        </div>

        {/* ── Per-LED Signal Cards (JS geometry) ── */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-4 flex flex-col gap-3">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <h4 className="font-bold text-xs text-slate-200">Per-LED Signal Gain H(0)</h4>
          </div>
          <div className="flex flex-col gap-2">
            {state.leds.map((led) => (
              <LEDSignalCard
                key={led.id}
                led={led}
                dist={state.distances[led.id] || 0}
                gain={state.dcGains[led.id] || 0}
                isLos={!!state.losMatrix[led.id]}
                isFov={!!state.visibilityMatrix[led.id]}
              />
            ))}
          </div>
        </div>

        {/* ── Physics Engine Metrics (Python backend) ── */}
        <PhysicsPanel metrics={physicsMetrics} loading={physicsLoading} />

        {/* ── Communication Engine Metrics (Python backend — Module 3) ── */}
        <CommunicationPanel metrics={commMetrics} loading={commLoading} />

        {/* ── Localization Engine Metrics (Python backend — Module 4) ── */}
        {state.leds.length > 4 ? (
          <MultiLightPanel metrics={multicellMetrics ?? null} loading={!!multicellLoading} numLeds={state.leds.length} />
        ) : (
          <LocalizationPanel metrics={localizationMetrics} loading={localizationLoading} />
        )}

        {/* ── Integrated Engine Metrics (Python backend — Module 5) ── */}
        <IntegratedPanel metrics={integratedMetrics} loading={integratedLoading} />

        {/* ── Adaptive Modulation Engine Metrics (Python backend — Module 6) ── */}
        <AdaptivePanel metrics={adaptiveMetrics} loading={adaptiveLoading} />

        {/* ── Power & Pre-Equalization Engine Metrics (Python backend — Module 7) ── */}
        <PowerPanel metrics={powerMetrics} loading={powerLoading} />

        {/* ── Joint Optimization Engine Metrics (Python backend — Module 8) ── */}
        <JointPanel metrics={jointMetrics} isLoading={jointLoading} />

      </div>
    </div>
  );
}

