import { useMemo, useState } from "react";
import { ChevronDown, Download, FileJson, FileSpreadsheet, Pause, Play, Search, Settings2 } from "lucide-react";
import type { SimulationState } from "../../types";
import { workbookBlob } from "./xlsx";
import type { LiveResultsState, ResultRow, SheetName } from "./types";

const SHEETS: Array<[SheetName, string]> = [["overview", "Overview"], ["environment", "Environment"], ["optical_channel", "Optical Channel"], ["communication", "Communication"], ["localization", "Localization"], ["subcarriers", "Subcarriers"], ["power", "Power"], ["optimization", "Optimization"], ["validation", "Validation"], ["events", "Events"]];
const label = (name: string) => name.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
const display = (value: unknown) => value === null || value === undefined ? "N/A" : (typeof value === "number" && Number.isNaN(value)) ? "N/A" : typeof value === "boolean" ? (value ? "TRUE" : "FALSE") : typeof value === "number" ? (Math.abs(value) > 0 && (Math.abs(value) < .001 || Math.abs(value) >= 1e6) ? value.toExponential(3) : value.toFixed(Math.abs(value) < 1 ? 4 : 3)) : String(value);
const download = (blob: Blob, filename: string) => { const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = filename; link.click(); URL.revokeObjectURL(link.href); };

interface Props { state: SimulationState; telemetry: LiveResultsState; }

export default function LiveResults({ state, telemetry }: Props) {
  const [sheet, setSheet] = useState<SheetName>("overview");
  const [search, setSearch] = useState("");
  const [device, setDevice] = useState("ALL");
  const [paused, setPaused] = useState(false);
  const [followLatest, setFollowLatest] = useState(true);
  const [sort, setSort] = useState<{ key: string; direction: 1 | -1 } | null>({ key: "frame_id", direction: -1 });
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [showColumns, setShowColumns] = useState(false);
  const [selected, setSelected] = useState<ResultRow | null>(null);
  const sourceRows = telemetry.rows[sheet] ?? [];
  const allColumns = useMemo(() => [...new Set(sourceRows.flatMap((row) => Object.keys(row)))], [sourceRows]);
  const filtered = useMemo(() => sourceRows.filter((row) => {
    if (device !== "ALL" && row.device_id !== device) return false;
    return !search || Object.values(row).some((value) => String(value ?? "").toLowerCase().includes(search.toLowerCase()));
  }).sort((a, b) => !sort ? 0 : ((a[sort.key] ?? "") > (b[sort.key] ?? "") ? sort.direction : (a[sort.key] ?? "") < (b[sort.key] ?? "") ? -sort.direction : 0)), [sourceRows, search, device, sort]);
  const visibleColumns = allColumns.filter((column) => !hidden.has(column));
  const shownRows = paused ? filtered.slice(0, 200) : filtered.slice(0, 200);
  const summary = telemetry.rows.overview?.at(-1);
  const devices = [...new Set((telemetry.rows.overview ?? []).map((row) => String(row.device_id)).filter(Boolean))];
  const snapshot = async () => {
    const response = await fetch(`/api/results/${telemetry.runId}/export`);
    return response.ok ? await response.json() : { run_id: telemetry.runId, frame_id: telemetry.latestFrame, rows: telemetry.rows, metadata: {} };
  };
  const exportXlsx = async () => { const data = await snapshot(); download(workbookBlob(data.rows, { ...data.metadata, export_timestamp: new Date().toISOString(), snapshot_frame_id: data.frame_id }), `VLCL_Results_${telemetry.runId}_Frame_${data.frame_id}.xlsx`); };
  const exportJson = async () => { const data = await snapshot(); download(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }), `VLCL_Results_${telemetry.runId}_Frame_${data.frame_id}.json`); };
  const exportCsv = () => { const data = [visibleColumns.join(","), ...filtered.map((row) => visibleColumns.map((column) => JSON.stringify(row[column] ?? "")).join(","))].join("\n"); download(new Blob([data], { type: "text/csv" }), `VLCL_${sheet}_${telemetry.latestFrame}.csv`); };

  return <section className="flex-1 min-h-0 overflow-hidden rounded-2xl border border-slate-800 bg-[#080f1e] shadow-2xl flex flex-col">
    <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3 border-b border-slate-800 bg-slate-950/50">
      <div><h2 className="text-sm font-black tracking-wide text-white">LIVE RESULTS <span className="ml-2 text-[10px] text-emerald-400 badge-live">SIMULATION {state.isPlaying ? "RUNNING" : "PAUSED"}</span></h2><p className="text-[10px] text-slate-500 mt-0.5">Authoritative frames are committed server-side · schema v1.0.0</p></div>
      <div className="flex items-center gap-4 font-mono text-[11px]"><span className="text-slate-400">Frame <b className="text-cyan-300">{telemetry.latestFrame.toLocaleString()}</b></span><span className="text-slate-400">Time <b className="text-white">{state.currentTime.toFixed(2)}s</b></span><span className="text-slate-400">Sum Rate <b className="text-white">{display(summary?.sum_rate_mbps)} Mbps</b></span><span className="text-emerald-400">● QoS {summary?.qos_status === false ? "FAIL" : "OK"}</span></div>
    </div>
    <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 border-b border-slate-800 bg-[#0b1324]">
      {[['Current Frame', telemetry.latestFrame.toLocaleString()], ['Simulation Time', `${state.currentTime.toFixed(2)}s`], ['Sum Rate', `${display(summary?.sum_rate_mbps)} Mbps`], ['Loc Error', `${display(summary?.localization_error_m)} m`], ['SNR', `${display(summary?.snr_db)} dB`], ['BER', display(summary?.ber)], ['Power', `${display(summary?.total_power_w)} W`], ['Status', summary?.overall_status === false ? 'FAIL' : 'QoS OK']].map(([name, item]) => <div key={name} className="px-4 py-2 border-r border-slate-800"><p className="text-[9px] uppercase tracking-wider text-slate-500">{name}</p><p className="text-xs font-bold font-mono text-slate-200 truncate">{item}</p></div>)}
    </div>
    <div className="flex flex-wrap gap-2 items-center px-4 py-2 border-b border-slate-800 bg-slate-950/30">
      <label className="flex items-center gap-2 min-w-[180px] rounded-lg border border-slate-700 bg-slate-900 px-2 py-1.5"><Search className="w-3.5 h-3.5 text-slate-500"/><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search rows" className="w-full bg-transparent outline-none text-xs text-slate-200 placeholder:text-slate-600"/></label>
      <label className="relative"><select value={device} onChange={(event) => setDevice(event.target.value)} className="appearance-none rounded-lg border border-slate-700 bg-slate-900 py-1.5 pl-2 pr-7 text-xs text-slate-300"><option>ALL</option>{devices.map((item) => <option key={item}>{item}</option>)}</select><ChevronDown className="absolute right-2 top-2 w-3 h-3 text-slate-500 pointer-events-none"/></label>
      <button onClick={() => setPaused(!paused)} className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900 px-2.5 py-1.5 text-xs font-semibold text-slate-300 hover:bg-slate-800">{paused ? <Play className="w-3 h-3"/> : <Pause className="w-3 h-3"/>}{paused ? "Resume View" : "Pause View"}</button>
      <button onClick={() => setFollowLatest(!followLatest)} className={`rounded-lg border px-2.5 py-1.5 text-xs font-semibold ${followLatest ? "border-cyan-700 bg-cyan-950/50 text-cyan-300" : "border-slate-700 bg-slate-900 text-slate-400"}`}>Following Latest</button>
      <div className="relative"><button onClick={() => setShowColumns(!showColumns)} className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900 px-2.5 py-1.5 text-xs text-slate-300"><Settings2 className="w-3 h-3"/>Columns</button>{showColumns && <div className="absolute z-20 top-9 left-0 max-h-56 w-56 overflow-auto rounded-xl border border-slate-700 bg-slate-950 p-2 shadow-2xl">{allColumns.map((column) => <label className="flex gap-2 p-1 text-[11px] text-slate-300" key={column}><input type="checkbox" checked={!hidden.has(column)} onChange={() => setHidden((current) => { const next = new Set(current); next.has(column) ? next.delete(column) : next.add(column); return next; })}/>{label(column)}</label>)}</div>}</div>
      <div className="ml-auto flex gap-2"><button onClick={exportCsv} className="flex items-center gap-1 rounded-lg border border-slate-700 px-2.5 py-1.5 text-xs text-slate-300 hover:bg-slate-800"><Download className="w-3 h-3"/>CSV</button><button onClick={exportJson} className="flex items-center gap-1 rounded-lg border border-slate-700 px-2.5 py-1.5 text-xs text-slate-300 hover:bg-slate-800"><FileJson className="w-3 h-3"/>JSON</button><button onClick={exportXlsx} className="flex items-center gap-1 rounded-lg border border-cyan-700 bg-cyan-950/50 px-2.5 py-1.5 text-xs font-bold text-cyan-300 hover:bg-cyan-900/50"><FileSpreadsheet className="w-3 h-3"/>Quick XLSX</button></div>
    </div>
    <div className="min-h-0 flex-1 overflow-auto custom-scrollbar"><table className="min-w-full border-separate border-spacing-0 font-mono text-[11px]"><thead className="sticky top-0 z-10 bg-[#101a2d]"> <tr>{visibleColumns.map((column) => <th key={column} onClick={() => setSort((old) => old?.key === column ? { key: column, direction: old.direction === 1 ? -1 : 1 } : { key: column, direction: 1 })} className="cursor-pointer whitespace-nowrap border-b border-r border-slate-700 px-3 py-2 text-left text-[10px] uppercase tracking-wider text-slate-400 hover:text-cyan-300">{label(column)} {sort?.key === column ? (sort.direction === 1 ? '↑' : '↓') : ''}</th>)}</tr></thead><tbody>{shownRows.map((row, index) => <tr key={`${row.frame_id}-${row.device_id}-${index}`} onClick={() => setSelected(row)} className={`cursor-pointer hover:bg-cyan-950/30 ${row.frame_id === telemetry.latestFrame ? "bg-cyan-950/20" : ""}`}>{visibleColumns.map((column) => <td key={column} className={`whitespace-nowrap border-b border-r border-slate-800 px-3 py-2 ${typeof row[column] === "boolean" ? (row[column] ? "text-emerald-400" : "text-rose-400") : "text-slate-300"}`}>{display(row[column])}</td>)}</tr>)}{!shownRows.length && <tr><td colSpan={Math.max(visibleColumns.length, 1)} className="p-8 text-center text-slate-500">No committed {label(sheet)} rows yet.</td></tr>}</tbody></table></div>
    <div className="flex flex-wrap items-center gap-x-5 gap-y-1 border-t border-slate-800 bg-slate-950/50 px-4 py-2 text-[10px] text-slate-500"><span>Rows: {sourceRows.length.toLocaleString()}</span><span>Frames: {telemetry.latestFrame.toLocaleString()}</span><span className={telemetry.pendingFrames ? "text-amber-400" : "text-emerald-400"}>{telemetry.pendingFrames ? `${telemetry.pendingFrames} committing` : "● Committed"}</span><span>Last committed: {telemetry.latestFrame.toLocaleString()}</span>{telemetry.commitError && <span className="text-rose-400">Store: {telemetry.commitError}</span>}</div>
    <div className="flex overflow-x-auto border-t border-slate-800 bg-[#0b1324]">{SHEETS.map(([key, name]) => <button onClick={() => { setSheet(key); setSelected(null); }} key={key} className={`whitespace-nowrap border-r border-slate-800 px-3 py-2 text-[10px] font-bold ${sheet === key ? "bg-cyan-500 text-slate-950" : "text-slate-400 hover:bg-slate-800"}`}>{name}</button>)}</div>
    {selected && <aside className="absolute bottom-6 right-6 z-30 w-80 rounded-xl border border-cyan-800 bg-[#0b1324] p-4 shadow-2xl"><div className="mb-2 flex justify-between"><b className="text-xs text-cyan-300">Frame Inspector</b><button onClick={() => setSelected(null)} className="text-slate-500">×</button></div><div className="max-h-60 overflow-auto font-mono text-[10px]">{Object.entries(selected).map(([key, item]) => <div key={key} className="flex justify-between gap-3 border-b border-slate-800 py-1"><span className="text-slate-500">{label(key)}</span><span className="text-slate-200 text-right">{display(item)}</span></div>)}</div></aside>}
  </section>;
}
