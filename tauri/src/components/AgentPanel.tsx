import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";

interface TaskEntry {
  id: string;
  cmd: string;
  output: string;
  error: string;
  time: string;
  status: string;
}

type AgentProvider = "qwen" | "koboldcpp" | "github_sdk";

const PROVIDER_MODEL_PRESETS: Record<AgentProvider, Array<{ value: string; label: string }>> = {
  qwen: [
    { value: "qwen3.6-plus", label: "Qwen 3.6 Plus" },
    { value: "qwen-plus", label: "Qwen Plus" },
    { value: "qwen-turbo", label: "Qwen Turbo" },
  ],
  koboldcpp: [
    { value: "DeepSeek-R1-Distill-Qwen-7B", label: "DeepSeek-R1 Distill Qwen 7B (free, reasoned)" },
    { value: "Qwen2.5-7B-Instruct-Q4_K_M", label: "Qwen2.5 7B Instruct" },
    { value: "Qwen2.5-14B-Instruct-Q4_K_M", label: "Qwen2.5 14B Instruct" },
    { value: "Mistral-7B-Instruct-v0.3", label: "Mistral 7B Instruct" },
  ],
  github_sdk: [
    { value: "gpt-4.1", label: "GPT-4.1" },
    { value: "gpt-4.1-mini", label: "GPT-4.1 Mini" },
    { value: "o4-mini", label: "o4-mini" },
  ],
};

export default function AgentPanel() {
  const [request, setRequest] = useState("");
  const [output, setOutput] = useState("");
  const [running, setRunning] = useState(false);
  const [tasks, setTasks] = useState<TaskEntry[]>([]);
  const [workDir, setWorkDir] = useState("");
  const [provider, setProvider] = useState<AgentProvider>("qwen");
  const [modelPreset, setModelPreset] = useState<string>("custom");
  const [customModel, setCustomModel] = useState<string>("DeepSeek-R1-Distill-Qwen-7B");
  const [statusChecking, setStatusChecking] = useState(false);

  const effectiveModel = modelPreset === "custom" ? customModel.trim() : modelPreset;

  useEffect(() => {
    const saved = localStorage.getItem("agent_provider");
    if (saved === "qwen" || saved === "koboldcpp" || saved === "github_sdk") {
      setProvider(saved);
    }

    const savedPreset = localStorage.getItem("agent_model_preset");
    if (savedPreset) setModelPreset(savedPreset);

    const savedCustom = localStorage.getItem("agent_model_custom");
    if (savedCustom) setCustomModel(savedCustom);
  }, []);

  useEffect(() => {
    localStorage.setItem("agent_provider", provider);
  }, [provider]);

  useEffect(() => {
    localStorage.setItem("agent_model_preset", modelPreset);
  }, [modelPreset]);

  useEffect(() => {
    localStorage.setItem("agent_model_custom", customModel);
  }, [customModel]);

  useEffect(() => {
    const presets = PROVIDER_MODEL_PRESETS[provider];
    const isKnown = presets.some((p) => p.value === modelPreset);
    if (!isKnown && modelPreset !== "custom") {
      setModelPreset(presets[0]?.value ?? "custom");
    }
  }, [provider, modelPreset]);

  // Resolve work directory — ask Rust for the absolute path so it works
  // in both dev (tauri/) and release (tauri/src-tauri/target/release/) modes.
  useEffect(() => {
    invoke<string>("get_work_dir")
      .then((dir) => setWorkDir(dir))
      .catch(() => setWorkDir("../"));  // fallback for hot-reload dev
  }, []);

  const addTask = (cmd: string, out: string, err: string, status: string) => {
    const entry: TaskEntry = {
      id: `task-${Date.now()}`,
      cmd,
      output: out,
      error: err,
      time: new Date().toLocaleTimeString(),
      status,
    };
    setTasks((prev) => [entry, ...prev]);
  };

  const handleRunRequest = async () => {
    if (!request.trim()) return;
    setRunning(true);
    setOutput("");
    try {
      const result: any = await invoke("ai_direct_request", {
        request,
        workDir: workDir || "..",
        provider,
        modelOverride: effectiveModel || undefined,
      });
      setOutput(result.stdout || "");
      if (result.stderr) setOutput((prev) => prev + "\n\n--- stderr ---\n" + result.stderr);
      addTask(
        `[${provider}:${effectiveModel || "default"}] ai_director.py --request "${request.slice(0, 60)}..." --execute`,
        (result.stdout || "").slice(-200),
        (result.stderr || "").slice(-200),
        result.status
      );
    } catch (e: any) {
      setOutput(`Error: ${e}`);
      addTask(
        `[${provider}:${effectiveModel || "default"}] ai_director.py --request "${request.slice(0, 60)}..." --execute`,
        "",
        String(e),
        "error"
      );
    }
    setRunning(false);
  };

  const handleProviderStatus = async () => {
    setStatusChecking(true);
    try {
      const status = await invoke<string>("agent_provider_status", {
        provider,
        workDir: workDir || "..",
        modelOverride: effectiveModel || undefined,
      });
      setOutput(status);
      addTask(
        `Provider Status [${provider}:${effectiveModel || "default"}]`,
        status.slice(-200),
        "",
        "success"
      );
    } catch (e: any) {
      const msg = `Provider status error: ${String(e)}`;
      setOutput(msg);
      addTask(
        `Provider Status [${provider}:${effectiveModel || "default"}]`,
        "",
        String(e),
        "error"
      );
    }
    setStatusChecking(false);
  };

  const handleRunProbe = async () => {
    setRunning(true);
    setOutput("");
    try {
      const result: any = await invoke("run_background_probe", {
        workDir: workDir || "..",
      });
      setOutput(result.stdout || "");
      if (result.stderr) setOutput((prev) => prev + "\n\n--- stderr ---\n" + result.stderr);
      addTask("background_probe.py --once", (result.stdout || "").slice(-200), (result.stderr || "").slice(-200), result.status);
    } catch (e: any) {
      setOutput(`Error: ${e}`);
      addTask("background_probe.py --once", "", String(e), "error");
    }
    setRunning(false);
  };

  const handleCheckNodes = async () => {
    setRunning(true);
    setOutput("");
    try {
      const result: any = await invoke("check_nodes", { workDir: workDir || ".." });
      setOutput(result.stdout || "");
      if (result.stderr) setOutput((prev) => prev + "\n\n--- stderr ---\n" + result.stderr);
      addTask("orchestrator --status", (result.stdout || "").slice(-200), (result.stderr || "").slice(-200), result.status);
    } catch (e: any) {
      setOutput(`Error: ${e}`);
      addTask("orchestrator --status", "", String(e), "error");
    }
    setRunning(false);
  };

  const handleSmartDownload = async () => {
    setRunning(true);
    setOutput("🔍 Querying NASA Catalog via Rust SDK...");
    try {
      // 1. Ask Rust to query the real NASA CMR catalog
      const manifest: any = await invoke("search_nasa_granules", {
        workDir: workDir || '../',
        bbox: [44.5, -92.0, 47.0, -80.0],
        startDate: "2024-06-01",
        endDate: "2024-10-30",
        sensor: "hls,sar"
      });

      const count = manifest.granules?.length || 0;
      setOutput(`✅ Found ${count} granules! Starting Swarm Download...`);
      addTask(`NASA Search`, `Found ${count} granules. Starting download swarm...`, '', 'running');

      // 2. Trigger the Python Swarm via run_task.
      //    NOTE: Rust's run_task() parameter is named `cwd` (not work_dir) —
      //    pass it exactly as `cwd` so Tauri/serde doesn't rename it.
      const resolvedDir = workDir || '../';
      const res: any = await invoke("run_task", {
        taskId: `dl-swarm-${Date.now()}`,
        script: "batch_download_manager.py",
        args: ['--lakes', 'all', '--start', '2020', '--end', '2025', '--sensors', 'hls,sar'],
        cwd: resolvedDir,
      });
      
      setOutput(res.stdout || "");
      if (res.stderr) setOutput((prev) => prev + "\n\n--- stderr ---\n" + res.stderr);
      addTask("Swarm Download (2020-2025 all lakes)", (res.stdout || "").slice(-400), (res.stderr || "").slice(-200), res.status);
    } catch (e: any) {
      setOutput(`❌ Error: ${e}`);
      addTask("Smart Download", "", String(e), "error");
    }
    setRunning(false);
  };

  return (
    <div className="agent-panel" style={{ padding: 16, height: "100%", display: "flex", flexDirection: "column", gap: 12 }}>
      <h2 style={{ margin: 0 }}>🤖 AI Director</h2>
      <p style={{ color: "#888", fontSize: 13 }}>
        Toggle providers and model profiles like a mini model selector so you can A/B test agent behavior quickly.
      </p>

      {/* Work directory */}
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <label style={{ color: "#888", fontSize: 12 }}>Work dir:</label>
        <input
          value={workDir}
          onChange={(e) => setWorkDir(e.target.value)}
          style={{ flex: 1, padding: "4px 8px", borderRadius: 4, border: "1px solid #333", background: "#1a1a2e", color: "#fff", fontSize: 12 }}
          placeholder="Path to cesarops-core/"
        />
        <label style={{ color: "#888", fontSize: 12 }}>Provider:</label>
        <select
          value={provider}
          onChange={(e) => setProvider(e.target.value as AgentProvider)}
          style={{
            minWidth: 160,
            padding: "4px 8px",
            borderRadius: 4,
            border: "1px solid #333",
            background: "#1a1a2e",
            color: "#fff",
            fontSize: 12,
          }}
        >
          <option value="qwen">Qwen (DashScope)</option>
          <option value="koboldcpp">KoboldCpp (local)</option>
          <option value="github_sdk">GitHub SDK endpoint</option>
        </select>
        <button
          onClick={handleProviderStatus}
          disabled={running || statusChecking}
          style={{
            padding: "6px 10px",
            borderRadius: 6,
            border: "1px solid #333",
            background: "#1a1a2e",
            color: "#fff",
            cursor: statusChecking ? "not-allowed" : "pointer",
            fontSize: 11,
            opacity: statusChecking ? 0.6 : 1,
          }}
        >
          {statusChecking ? "Checking…" : "Provider Status"}
        </button>
      </div>

      {/* Model profile selector */}
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <label style={{ color: "#888", fontSize: 12, minWidth: 62 }}>Agent:</label>
        <select
          value={modelPreset}
          onChange={(e) => setModelPreset(e.target.value)}
          style={{
            minWidth: 280,
            padding: "4px 8px",
            borderRadius: 4,
            border: "1px solid #333",
            background: "#1a1a2e",
            color: "#fff",
            fontSize: 12,
          }}
        >
          {PROVIDER_MODEL_PRESETS[provider].map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
          <option value="custom">Custom model id...</option>
        </select>
        {modelPreset === "custom" && (
          <input
            value={customModel}
            onChange={(e) => setCustomModel(e.target.value)}
            placeholder="Enter model id"
            style={{
              flex: 1,
              padding: "4px 8px",
              borderRadius: 4,
              border: "1px solid #333",
              background: "#1a1a2e",
              color: "#fff",
              fontSize: 12,
            }}
          />
        )}
        <span style={{ color: "#8b949e", fontSize: 11 }}>
          Active: {effectiveModel || "default"}
        </span>
      </div>

      {/* Request input + action buttons */}
      <div style={{ display: "flex", gap: 8 }}>
        <textarea
          rows={3}
          style={{ flex: 1, padding: "8px 12px", borderRadius: 6, border: "1px solid #333", background: "#1a1a2e", color: "#fff", fontSize: 13, fontFamily: "inherit", resize: "vertical" }}
          placeholder='e.g. "Scan Straits of Mackinac east and west for anomalies, be aggressive"'
          value={request}
          onChange={(e) => setRequest(e.target.value)}
        />
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <button
            onClick={handleRunRequest}
            disabled={running || !request.trim()}
            style={{ padding: "8px 16px", borderRadius: 6, border: "none", background: "#4361ee", color: "#fff", cursor: "pointer", opacity: running ? 0.5 : 1 }}
          >
            {running ? "Running…" : "Run"}
          </button>
          <button onClick={handleRunProbe} disabled={running} style={{ padding: "6px 12px", borderRadius: 6, border: "1px solid #333", background: "#1a1a2e", color: "#fff", cursor: "pointer", fontSize: 11 }}>
            🔍 Probe
          </button>
          <button onClick={handleCheckNodes} disabled={running} style={{ padding: "6px 12px", borderRadius: 6, border: "1px solid #333", background: "#1a1a2e", color: "#fff", cursor: "pointer", fontSize: 11 }}>
            📡 Nodes
          </button>
        </div>
      </div>

      {/* ── Swarm Download — full-width row so it's never clipped ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <button
          onClick={handleSmartDownload}
          disabled={running}
          style={{
            padding: "10px 24px", borderRadius: 6, border: "none",
            background: running ? "#333" : "#059669",
            color: "#fff", cursor: running ? "not-allowed" : "pointer",
            fontSize: 13, fontWeight: "bold", flexShrink: 0,
          }}
        >
          📥 {running ? "Downloading…" : "Swarm Download"}
        </button>
        <span style={{ fontSize: 11, color: "#8b949e" }}>
          Searches NASA catalog then runs <code style={{ background: "#1a1a2e", padding: "1px 4px", borderRadius: 3 }}>batch_download_manager.py</code> — all lakes, 2020–2025, HLS+SAR
        </span>
      </div>

      {/* Output */}
      <div
        style={{
          flex: 1,
          overflow: "auto",
          padding: 12,
          borderRadius: 6,
          background: "#0d1117",
          fontFamily: "Consolas, 'Courier New', monospace",
          fontSize: 12,
          whiteSpace: "pre-wrap",
          color: "#c9d1d9",
          lineHeight: 1.5,
        }}
      >
        {output || (running ? "⏳ Running…" : "Output will appear here…")}
      </div>

      {/* Task log */}
      {tasks.length > 0 && (
        <div style={{ maxHeight: 150, overflow: "auto", borderRadius: 6, background: "#0d1117", padding: 8 }}>
          <div style={{ fontSize: 11, color: "#8b949e", marginBottom: 4 }}>Task Log</div>
          {tasks.map((t) => (
            <div key={t.id} style={{ fontSize: 11, borderBottom: "1px solid #21262d", padding: "3px 0", display: "flex", gap: 8 }}>
              <span style={{ color: "#8b949e", minWidth: 70 }}>{t.time}</span>
              <span style={{ color: t.status === "error" ? "#f85149" : "#3fb950", minWidth: 50 }}>{t.status}</span>
              <span style={{ color: "#c9d1d9", flex: 1 }}>{t.cmd}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
