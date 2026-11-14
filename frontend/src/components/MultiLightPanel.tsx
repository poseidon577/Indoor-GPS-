/**
 * MultiLightPanel.tsx
 *
 * Shown when the room has more than 4 LEDs. LEDs share frequency tones in four
 * colour groups, so one tone can arrive from several LEDs at once.
 * Compares two Python solvers (integrated_vlcl/multicell.py):
 *   - interference-aware: models every LED that shares a tone (the new method)
 *   - nearest light only: assumes each tone comes from the nearest LED (old assumption)
 */

import { Grid3x3, Loader, AlertCircle, CheckCircle2 } from "lucide-react";
import { MulticellMetrics, MulticellFix } from "../types";
import { REUSE_COLOURS_CSS } from "../lightLayout";

interface MultiLightPanelProps {
  metrics: MulticellMetrics | null;
  loading: boolean;
  numLeds: number;
}

function fmtErr(m: number | null) {
  if (m === null || m === undefined) return "—";
  return m < 1 ? `${(m * 100).toFixed(0)} cm` : `${m.toFixed(2)} m`;
}

function errClass(m: number | null) {
  if (m === null || m === undefined) return "text-slate-500";
  if (m < 0.3) return "text-emerald-400";
  if (m < 1.0) return "text-amber-400";
  return "text-rose-400";
}

function FixCard({ title, note, fix, accent }: { title: string; note: string; fix: MulticellFix; accent: string }) {
  const p = fix.estimated_position;
  return (
    <div className={`flex-1 rounded-xl border p-2.5 bg-slate-950/60 ${accent}`}>
      <p className="text-[10px] font-bold text-slate-200">{title}</p>
      <p className="text-[9px] text-slate-500 leading-snug mb-1.5">{note}</p>
      <p className={`font-mono font-bold text-lg leading-none ${errClass(fix.error_xy_m)}`} data-testid={`err-${title}`}>
        {fix.success ? fmtErr(fix.error_xy_m) : "failed"}
      </p>
      <p className="text-[9px] text-slate-500 mt-0.5">floor-plane error</p>
      {p && (
        <p className="font-mono text-[10px] text-slate-400 mt-1.5">
          est ({p[0].toFixed(2)}, {p[1].toFixed(2)})
        </p>
      )}
    </div>
  );
}

export default function MultiLightPanel({ metrics, loading, numLeds }: MultiLightPanelProps) {
  const cell = metrics?.nearest_cell.cell;
  return (
    <div className="bg-slate-900/50 border border-amber-900/40 rounded-2xl p-4 flex flex-col gap-3" data-testid="multilight-panel">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2">
          <Grid3x3 className="w-3.5 h-3.5 text-amber-300" />
          <h4 className="font-bold text-xs text-slate-200">Many-light Localization ({numLeds} LEDs)</h4>
        </div>
        {loading && (
          <span className="flex items-center gap-1 text-[10px] text-amber-300">
            <Loader className="w-3 h-3 animate-spin" /> Computing...
          </span>
        )}
        {!loading && !metrics && (
          <span className="flex items-center gap-1 text-[10px] text-slate-600">
            <AlertCircle className="w-3 h-3" /> Waiting for Python
          </span>
        )}
        {!loading && metrics && (
          <span className="flex items-center gap-1 text-[10px] text-emerald-400">
            <CheckCircle2 className="w-3 h-3" /> Live
          </span>
        )}
      </div>

      <p className="text-[10px] text-slate-500 leading-relaxed">
        Lights with the same colour send the same tones, so the receiver hears several of them mixed together.
        The left solver models that mixing. The right one is the old assumption that each tone comes from the nearest light.
      </p>

      {metrics && (
        <>
          <div className="flex gap-2">
            <FixCard
              title="Interference-aware"
              note="models every light sharing a tone"
              fix={metrics.interference_aware}
              accent="border-emerald-900/50"
            />
            <FixCard
              title="Nearest light only"
              note="old 4-light assumption"
              fix={metrics.nearest_cell}
              accent="border-rose-900/50"
            />
          </div>
          <div className="flex justify-between text-[10px] text-slate-500">
            <span>
              true ({metrics.true_position[0].toFixed(2)}, {metrics.true_position[1].toFixed(2)})
            </span>
            {cell && (
              <span>
                nearest-light cell: LEDs {Object.values(cell).join(", ")}
              </span>
            )}
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            {[1, 2, 3, 4].map((c) => (
              <span key={c} className="flex items-center gap-1 text-[10px] text-slate-400">
                <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: REUSE_COLOURS_CSS[c - 1] }} />
                {c === 1 ? "tones 1+5" : `tone ${c}`}
              </span>
            ))}
          </div>
          <p className="text-[9px] text-slate-600">
            The orange marker in the 3D view is the interference-aware estimate. Updates about every 3 s.
          </p>
        </>
      )}
    </div>
  );
}
