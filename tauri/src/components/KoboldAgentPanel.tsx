import React, { useState, useEffect, useCallback, useRef } from "react";
import { getApiBase } from "../services/api";

export default function KoboldAgentPanel() {
  const [koboldUrl, setKoboldUrl]           = useState("http://10.0.0.61:5001/v1");
  const [model, setModel]                   = useState("Qwen2.5-Coder-14B-Instruct");
  const [gpuMode, setGpuMode]               = useState("p100_dual");
  const [backend, setBackend]               = useState("cublas");
  const [reasoningModel, setReasoningModel] = useState("");
  const [huggingfaceModel, setHuggingfaceModel] = useState("");
  const [garmourPath, setGarmourPath]       = useState("/mnt/garmour/models");
  const [status, setStatus]                 = useState("");
  const [statusType, setStatusType]         = useState<"info" | "success" | "error" | "warn">("info");
  const [downloadJobId, setDownloadJobId]   = useState<string | null>(null);
  const [apiOnline, setApiOnline]           = useState<boolean | null>(null);
  const [koboldOnline, setKoboldOnline]     = useState<boolean | null>(null);
  const [watchdogStatus, setWatchdogStatus] = useState<{api_status?: string; kobold_status?: string; restarts?: {api: number; kobold: number}} | null>(null);
  const [autoStarting, setAutoStarting]     = useState(false);
  // File picker state
  const [localFile, setLocalFile]           = useState<File | null>(null);
  const [uploading, setUploading]           = useState(false);
  const fileInputRef                        = useRef<HTMLInputElement>(null);

  // ── Connection check ────────────────────────────────────────────────────────
  const checkConnections = useCallback(async () => {
    try {
      const apiBase = await getApiBase();
      const r = await fetch(`${apiBase}/kobold/status`, { signal: AbortSignal.timeout(3000) });
      if (r.ok) {
        const data = await r.json();
        setApiOnline(true);
        setKoboldOnline(data.online === true);
        if (data.online && data.kobold_url) {
          setKoboldUrl(data.kobold_url.replace(/\/+$/, "") + "/v1");
        }
      } else {
        setApiOnline(false);
        setKoboldOnline(false);
      }
    } catch {
      setApiOnline(false);
      setKoboldOnline(false);
    }
    // Also fetch watchdog state
    try {
      const apiBase = await getApiBase();
      const r = await fetch(`${apiBase}/watchdog/status`, { signal: AbortSignal.timeout(2000) });
      if (r.ok) setWatchdogStatus(await r.json());
    } catch { /* watchdog may not be running */ }
  }, []);

  // Auto-retry every 15s when offline
  useEffect(() => {
    checkConnections();
    const t = setInterval(() => {
      if (!apiOnline) checkConnections();
    }, 15000);
    return () => clearInterval(t);
  }, [checkConnections, apiOnline]);

  // ── Auto-start API via watchdog ─────────────────────────────────────────────
  const tryAutoStart = useCallback(async () => {
    setAutoStarting(true);
    setMsg("Attempting to start the API via watchdog…", "info");
    try {
      // Try the watchdog start endpoint — only works if watchdog itself is running
      const apiBase = await getApiBase();
      const r = await fetch(`${apiBase}/watchdog/start-api`, {
        method: "POST", signal: AbortSignal.timeout(5000),
      });
      if (r.ok) {
        setMsg("Watchdog triggered API start — rechecking in 5s…", "info");
        setTimeout(checkConnections, 5000);
      } else {
        setMsg("Watchdog not reachable. Run: bash scripts/deploy_services.sh", "warn");
      }
    } catch {
      setMsg("Could not reach watchdog. Start manually: bash scripts/deploy_services.sh", "warn");
    }
    setAutoStarting(false);
  }, [checkConnections]);

  // ── Download job polling ────────────────────────────────────────────────────
  useEffect(() => {
    if (!downloadJobId) return;
    const interval = setInterval(async () => {
      try {
        const apiBase = await getApiBase();
        const res = await fetch(`${apiBase}/kobold/download-status/${downloadJobId}`);
        if (res.ok) {
          const data = await res.json();
          if (data.status === "completed") {
            setMsg(`✓ Downloaded ${huggingfaceModel} to ${garmourPath}`, "success");
            setHuggingfaceModel("");
            setDownloadJobId(null);
          } else if (data.status === "failed") {
            setMsg(`Download failed: ${data.error || "unknown error"}`, "error");
            setDownloadJobId(null);
          } else {
            setMsg(`Downloading… (${data.status})`, "info");
          }
        }
      } catch { /* transient */ }
    }, 2000);
    return () => clearInterval(interval);
  }, [downloadJobId, huggingfaceModel, garmourPath]);

  // ── Helpers ─────────────────────────────────────────────────────────────────
  const setMsg = (text: string, type: "info" | "success" | "error" | "warn" = "info") => {
    setStatus(text);
    setStatusType(type);
  };

  // ── Launch ──────────────────────────────────────────────────────────────────
  const launch = async () => {
    setMsg("Launching agent…", "info");
    try {
      const apiBase = await getApiBase();
      const response = await fetch(`${apiBase}/kobold/launch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, gpu_mode: gpuMode, model_path: garmourPath, port: 5001, backend }),
      });
      if (response.ok) {
        const data = await response.json();
        if (data.status === "already_running") {
          setMsg(`✓ KoboldCPP already running — ${data.kobold_url}`, "success");
        } else if (data.status === "launching") {
          setMsg(`✓ Launching KoboldCPP (PID ${data.pid}) — poll /kobold/status to confirm`, "success");
          setKoboldUrl(data.kobold_url);
        } else {
          setMsg(data.message || "Launch initiated", "info");
        }
      } else {
        const err = await response.json().catch(() => ({}));
        setMsg(`Failed to launch: ${err.detail || response.statusText}`, "error");
      }
    } catch {
      setMsg("Connection error — is the wrecks API running on port 8099?", "error");
    }
  };

  // ── HuggingFace download ────────────────────────────────────────────────────
  const downloadModel = async () => {
    if (!huggingfaceModel) { setMsg("Please enter a HuggingFace model name", "warn"); return; }
    setMsg("Queuing download…", "info");
    try {
      const apiBase = await getApiBase();
      const response = await fetch(`${apiBase}/kobold/download-model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: huggingfaceModel, save_path: garmourPath }),
      });
      if (response.ok) {
        const data = await response.json();
        setDownloadJobId(data.job_id);
        setMsg(`Downloading ${huggingfaceModel}…`, "info");
      } else {
        const err = await response.json().catch(() => ({}));
        setMsg(`Download failed: ${err.detail || response.statusText}`, "error");
      }
    } catch {
      setMsg("Connection error — is the wrecks API running on port 8099?", "error");
    }
  };

  // ── Local file upload ───────────────────────────────────────────────────────
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    setLocalFile(file);
    if (file) setMsg(`Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`, "info");
  };

  const uploadLocalModel = async () => {
    if (!localFile) { setMsg("No file selected", "warn"); return; }

    const ext = localFile.name.split(".").pop()?.toLowerCase();
    if (ext !== "gguf" && ext !== "onnx") {
      setMsg("Only .gguf and .onnx files are supported", "warn");
      return;
    }

    setUploading(true);
    setMsg(`Uploading ${localFile.name}…`, "info");

    try {
      const apiBase = await getApiBase();
      const formData = new FormData();
      formData.append("file", localFile);
      formData.append("save_path", garmourPath);

      const response = await fetch(`${apiBase}/kobold/upload-model`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setMsg(`✓ Uploaded to ${data.saved_path}`, "success");
        setLocalFile(null);
        if (fileInputRef.current) fileInputRef.current.value = "";
      } else {
        const err = await response.json().catch(() => ({}));
        setMsg(`Upload failed: ${err.detail || response.statusText}`, "error");
      }
    } catch {
      setMsg("Upload error — is the wrecks API running on port 8099?", "error");
    } finally {
      setUploading(false);
    }
  };

  const getRecommendedModels = () => {
    if (gpuMode === "p100_dual") {
      return [
        "Qwen2.5-Coder-14B-Instruct",
        "DeepSeek-Coder-V2-Lite-Instruct",
        "CodeLlama-13B-Instruct",
      ];
    } else if (gpuMode === "mixed_p100_1070") {
      // P1000 4GB — small routing/doorbell models only
      return [
        "Phi-3-mini-4k-instruct",
        "Qwen2.5-Coder-1.5B-Instruct",
        "TinyLlama-1.1B",
      ];
    }
    // p100_single
    return [
      "Qwen2.5-Coder-7B-Instruct",
      "DeepSeek-R1-Distill-Qwen-7B",
      "CodeLlama-7B-Instruct",
      "StarCoder2-7B",
    ];
  };

  // ── Status box colours (dark-theme safe) ────────────────────────────────────
  const statusStyles: Record<string, React.CSSProperties> = {
    success: { background: "#0d3321", border: "1px solid #3fb950", color: "#3fb950" },
    error:   { background: "#3d1e1e", border: "1px solid #f85149", color: "#f85149" },
    warn:    { background: "#2d2200", border: "1px solid #d29922", color: "#d29922" },
    info:    { background: "#0d1f38", border: "1px solid #58a6ff", color: "#79c0ff" },
  };

  // Connection bar colours
  const connBg    = apiOnline === null ? "#1c2333" : apiOnline ? "#0d3321" : "#3d1e1e";
  const connBdr   = apiOnline === null ? "#30363d" : apiOnline ? "#3fb950" : "#f85149";
  const connColor = apiOnline === null ? "#8b949e" : apiOnline ? "#3fb950" : "#f85149";

  // Offline hint colours
  const hintStyle: React.CSSProperties = {
    padding: "10px 12px", borderRadius: 6, marginBottom: 12,
    background: "#2d2200", border: "1px solid #d29922",
    color: "#d29922", fontSize: 12,
  };

  return (
    <div className="panel" style={{ color: "var(--text)" }}>
      <h2 style={{ color: "var(--text)", marginBottom: 12 }}>🤖 Coding Agent (Kobold)</h2>

      {/* ── Connection status bar ─────────────────────────────────────────── */}
      <div style={{
        display: "flex", alignItems: "center", gap: "1rem",
        padding: "8px 12px", borderRadius: 6, marginBottom: 12,
        background: connBg, border: `1px solid ${connBdr}`,
        color: connColor, fontSize: 13,
      }}>
        <span>
          <strong>API (8099):</strong>{" "}
          {apiOnline === null ? "⏳ checking…" : apiOnline ? "✓ Online" : "✗ Offline"}
        </span>
        <span>
          <strong>KoboldCPP (5001):</strong>{" "}
          {koboldOnline === null ? "⏳ checking…" : koboldOnline ? "✓ Running" : "✗ Not running"}
        </span>
        <button
          onClick={checkConnections}
          style={{
            marginLeft: "auto", padding: "3px 10px", borderRadius: 4,
            border: "1px solid #30363d", background: "#161b22",
            color: "#c9d1d9", cursor: "pointer", fontSize: 12,
          }}
        >
          🔄 Recheck
        </button>
      </div>

      {/* ── Offline hint ──────────────────────────────────────────────────── */}
      {apiOnline === false && (
        <div style={hintStyle}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
            <strong>Backend not reachable.</strong>
            <button
              onClick={tryAutoStart}
              disabled={autoStarting}
              style={{
                padding: "4px 12px", borderRadius: 4, border: "1px solid #d29922",
                background: "#1a1400", color: "#d29922", cursor: autoStarting ? "not-allowed" : "pointer",
                fontSize: 12, opacity: autoStarting ? 0.6 : 1,
              }}
            >
              {autoStarting ? "⏳ Starting…" : "⚡ Auto-start"}
            </button>
          </div>
          <div style={{ marginBottom: 6, opacity: 0.9 }}>
            Or start manually on the Linux node:
          </div>
          <code style={{
            display: "block", padding: "4px 8px",
            background: "#0d1117", color: "#7ec8e3", borderRadius: 4, marginBottom: 4,
          }}>
            bash scripts/deploy_services.sh
          </code>
          <div style={{ opacity: 0.7, fontSize: 11 }}>
            Installs systemd services that start on boot and restart on crash.
            Check status: <code style={{ color: "#d29922" }}>bash scripts/deploy_services.sh --status</code>
          </div>
        </div>
      )}

      {/* ── Watchdog restart counter ───────────────────────────────────────── */}
      {watchdogStatus && (watchdogStatus.restarts?.api ?? 0) + (watchdogStatus.restarts?.kobold ?? 0) > 0 && (
        <div style={{
          padding: "6px 10px", borderRadius: 6, marginBottom: 8,
          background: "#1a1400", border: "1px solid #d29922",
          color: "#d29922", fontSize: 11,
        }}>
          ⚠ Watchdog has restarted services —
          API: {watchdogStatus.restarts?.api ?? 0}×,
          KoboldCPP: {watchdogStatus.restarts?.kobold ?? 0}×
        </div>
      )}

      {/* ── Inference URL ─────────────────────────────────────────────────── */}
      <div className="form-group">
        <label style={{ color: "var(--text-dim)" }}>API Base URL:</label>
        <input
          value={koboldUrl}
          onChange={e => setKoboldUrl(e.target.value)}
          placeholder="http://localhost:5001/v1"
        />
        <small style={{ color: "var(--text-dim)" }}>Use this URL in Copilot / Roo Code settings</small>
      </div>

      {/* ── GPU mode ──────────────────────────────────────────────────────── */}
      <div className="form-group">
        <label style={{ color: "var(--text-dim)" }}>Node / GPU:</label>
        <select value={gpuMode} onChange={e => setGpuMode(e.target.value)}>
          <option value="p100_dual">T440 — Dual P100 16GB (32GB total, 14B models)</option>
          <option value="p100_single">T440 — Single P100 16GB (7B models)</option>
          <option value="mixed_p100_1070">i7 — P1000 4GB (small/routing models)</option>
        </select>
      </div>

      {/* ── Compute backend ───────────────────────────────────────────────── */}
      <div className="form-group">
        <label style={{ color: "var(--text-dim)" }}>Compute Backend:</label>
        <select value={backend} onChange={e => setBackend(e.target.value)}>
          <option value="cublas">CUDA / cuBLAS (P100 — recommended)</option>
          <option value="vulkan">Vulkan / wgpu (fallback, any GPU)</option>
          <option value="cpu">CPU only</option>
        </select>
        <small style={{ color: "var(--text-dim)" }}>
          cuBLAS = fastest on P100. Vulkan = wgpu-compatible, works without CUDA drivers.
        </small>
      </div>

      {gpuMode === "mixed_p100_1070" && (
        <div className="form-group">
          <label style={{ color: "var(--text-dim)" }}>Reasoning Model (Secondary GPU):</label>
          <select value={reasoningModel} onChange={e => setReasoningModel(e.target.value)}>
            <option value="">None — keep secondary free</option>
            <option value="phi-3-mini">Phi-3-Mini (1.8B)</option>
            <option value="tinyllama">TinyLlama (1.1B)</option>
            <option value="qwen1.5-0.5b">Qwen1.5-0.5B</option>
            <option value="starcoder2-3b">StarCoder2-3B</option>
          </select>
        </div>
      )}

      {/* ── Model selector ────────────────────────────────────────────────── */}
      <div className="form-group">
        <label style={{ color: "var(--text-dim)" }}>Model:</label>
        <select value={model} onChange={e => setModel(e.target.value)}>
          {getRecommendedModels().map(m => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      {/* ── Garmour path ──────────────────────────────────────────────────── */}
      <div className="form-group">
        <label style={{ color: "var(--text-dim)" }}>Garmour Model Path:</label>
        <input
          value={garmourPath}
          onChange={e => setGarmourPath(e.target.value)}
          placeholder="/mnt/garmour/models"
        />
      </div>

      {/* ── Upload local model ────────────────────────────────────────────── */}
      <div className="form-group">
        <label style={{ color: "var(--text-dim)" }}>Upload Local Model (.gguf / .onnx):</label>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          {/* Hidden real file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".gguf,.onnx"
            onChange={handleFileChange}
            style={{ display: "none" }}
            id="kobold-file-input"
          />
          {/* Styled trigger button */}
          <label
            htmlFor="kobold-file-input"
            style={{
              padding: "6px 14px", borderRadius: 6, cursor: "pointer",
              border: "1px solid #30363d", background: "#161b22",
              color: "#c9d1d9", fontSize: 13, whiteSpace: "nowrap",
              display: "inline-block",
            }}
          >
            📂 Choose File
          </label>
          {/* File name display */}
          <span style={{
            flex: 1, fontSize: 12, color: localFile ? "#c9d1d9" : "#6e7681",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {localFile ? `${localFile.name} (${(localFile.size / 1024 / 1024).toFixed(1)} MB)` : "No file chosen"}
          </span>
          {/* Upload button */}
          <button
            onClick={uploadLocalModel}
            disabled={!localFile || uploading}
            style={{
              padding: "6px 14px", borderRadius: 6,
              border: "none", cursor: localFile && !uploading ? "pointer" : "not-allowed",
              background: localFile && !uploading ? "#1f6feb" : "#21262d",
              color: localFile && !uploading ? "#fff" : "#6e7681",
              fontSize: 13, whiteSpace: "nowrap",
            }}
          >
            {uploading ? "⏳ Uploading…" : "⬆ Upload"}
          </button>
        </div>
        <small style={{ color: "var(--text-dim)" }}>
          Copies the file to {garmourPath} on the server via the API
        </small>
      </div>

      {/* ── HuggingFace download ──────────────────────────────────────────── */}
      <div className="form-group">
        <label style={{ color: "var(--text-dim)" }}>Download from HuggingFace:</label>
        <div style={{ display: "flex", gap: 8 }}>
          <input
            value={huggingfaceModel}
            onChange={e => setHuggingfaceModel(e.target.value)}
            placeholder="username/model-name"
            style={{ flex: 1 }}
          />
          <button onClick={downloadModel}>Download</button>
        </div>
      </div>

      {/* ── Launch ────────────────────────────────────────────────────────── */}
      <button onClick={launch} style={{ marginTop: 16 }}>
        Launch Agent
      </button>

      {/* ── Status message ────────────────────────────────────────────────── */}
      {status && (
        <div style={{
          marginTop: 12, padding: "8px 12px", borderRadius: 6,
          fontSize: 13, lineHeight: 1.5,
          ...statusStyles[statusType],
        }}>
          {status}
        </div>
      )}

      <p style={{ marginTop: 16, fontSize: "0.9em", color: "var(--text-dim)" }}>
        Compatible with Copilot / Roo Code. Models saved to {garmourPath}
      </p>
    </div>
  );
}
