import React, { useState, useEffect, useRef } from "react";
import { Terminal, Save, Play, Database, FileText, CheckCircle2, AlertCircle } from "lucide-react";

export default function DatasetPanel() {
  const [configContent, setConfigContent] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "success" | "error">("idle");
  
  const [isGenerating, setIsGenerating] = useState(false);
  const [terminalLogs, setTerminalLogs] = useState("");
  const terminalEndRef = useRef<HTMLDivElement>(null);

  // Load config on mount
  useEffect(() => {
    fetch("/api/dataset/config")
      .then(res => res.text())
      .then(data => setConfigContent(data))
      .catch(err => console.error("Failed to load dataset config:", err));
  }, []);

  // Auto-scroll terminal
  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [terminalLogs]);

  const handleSaveConfig = () => {
    setIsSaving(true);
    setSaveStatus("saving");
    fetch("/api/dataset/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: configContent })
    })
      .then(res => res.json())
      .then(data => {
        setIsSaving(false);
        setSaveStatus(data.success ? "success" : "error");
        setTimeout(() => setSaveStatus("idle"), 3000);
      })
      .catch(() => {
        setIsSaving(false);
        setSaveStatus("error");
        setTimeout(() => setSaveStatus("idle"), 3000);
      });
  };

  const handleGenerate = () => {
    if (isGenerating) return;
    
    // Save config first
    handleSaveConfig();
    
    setIsGenerating(true);
    setTerminalLogs(prev => prev + "\n\n> ==================================================\n> Starting Dataset Generation Process...\n> ==================================================\n\n");
    
    const eventSource = new EventSource("/api/dataset/generate-stream");
    
    eventSource.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "stdout" || data.type === "stderr") {
          // Remove carriage returns and formatting artifacts sometimes produced by tqdm over streams
          let text = data.text;
          setTerminalLogs(prev => prev + text);
        } else if (data.type === "close") {
          eventSource.close();
          setIsGenerating(false);
          setTerminalLogs(prev => prev + `\n\n> Process completed with exit code: ${data.code}\n`);
        }
      } catch (err) {
        console.error("Failed to parse SSE data", err);
      }
    };
    
    eventSource.onerror = () => {
      eventSource.close();
      setIsGenerating(false);
      setTerminalLogs(prev => prev + "\n> Error: Connection to generation process lost.\n");
    };
  };

  return (
    <div className="flex w-full h-full gap-4">
      
      {/* Left: Configuration Panel */}
      <div className="w-1/2 flex flex-col h-full bg-[#0b1120] border border-slate-800 rounded-2xl shadow-xl overflow-hidden">
        <div className="flex justify-between items-center px-5 py-3 border-b border-slate-800 bg-slate-900/50">
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-slate-200">dataset_config.yaml</h3>
          </div>
          <button
            onClick={handleSaveConfig}
            disabled={isSaving}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
          >
            {saveStatus === "saving" ? (
              <span className="animate-pulse">Saving...</span>
            ) : saveStatus === "success" ? (
              <><CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> Saved</>
            ) : saveStatus === "error" ? (
              <><AlertCircle className="w-3.5 h-3.5 text-rose-400" /> Error</>
            ) : (
              <><Save className="w-3.5 h-3.5 text-cyan-400" /> Save Config</>
            )}
          </button>
        </div>
        
        <div className="flex-1 p-0 overflow-hidden">
          <textarea
            value={configContent}
            onChange={(e) => setConfigContent(e.target.value)}
            className="w-full h-full bg-[#080c14] text-slate-300 font-mono text-[11px] p-4 resize-none outline-none focus:ring-1 ring-cyan-500/50 custom-scrollbar whitespace-pre"
            spellCheck="false"
          />
        </div>
      </div>

      {/* Right: Actions & Terminal */}
      <div className="w-1/2 flex flex-col h-full gap-4">
        
        {/* Action Card */}
        <div className="bg-[#0b1120] border border-slate-800 rounded-2xl p-5 shadow-xl flex-shrink-0 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-black text-white flex items-center gap-2 mb-1">
              <Database className="w-5 h-5 text-cyan-400" />
              Dataset Factory
            </h2>
            <p className="text-xs text-slate-400">Generate ML-ready Parquet datasets from the simulator.</p>
          </div>
          
          <button
            onClick={handleGenerate}
            disabled={isGenerating}
            className={`flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-bold transition shadow-lg ${
              isGenerating 
                ? "bg-slate-800 text-slate-500 border border-slate-700 cursor-not-allowed"
                : "bg-cyan-500 hover:bg-cyan-400 text-slate-950 border border-cyan-400 shadow-cyan-500/20"
            }`}
          >
            {isGenerating ? (
              <>
                <div className="w-4 h-4 rounded-full border-2 border-slate-500 border-t-transparent animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run Generation
              </>
            )}
          </button>
        </div>

        {/* Terminal Card */}
        <div className="flex-1 min-h-0 bg-[#080f1e] border border-slate-800 rounded-2xl shadow-xl flex flex-col overflow-hidden">
          <div className="flex justify-between items-center border-b border-slate-800 px-4 py-3 bg-slate-900/60">
            <div className="flex items-center gap-3">
              <Terminal className="w-4 h-4 text-slate-400" />
              <span className="text-slate-400 text-xs font-bold tracking-wider uppercase">Generation Console</span>
            </div>
            <button
              onClick={() => setTerminalLogs("")}
              className="px-2.5 py-1 bg-slate-800 border border-slate-700 rounded-lg text-[10px] hover:bg-slate-700 transition text-slate-400"
            >
              Clear
            </button>
          </div>
          <div className="flex-1 overflow-y-auto custom-scrollbar p-5">
            <pre className="text-cyan-400 text-xs leading-relaxed font-mono whitespace-pre-wrap break-all">
              {terminalLogs || "$ Waiting to start dataset generation...\n$ Edit the config on the left and hit 'Run Generation'."}
              <div ref={terminalEndRef} />
            </pre>
          </div>
        </div>
      </div>

    </div>
  );
}
