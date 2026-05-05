/**
 * SearchPanel — Primary mission entry point.
 *
 * The user types what they're looking for in plain English.
 * The backend parses intent, builds a mission spec, and queues it to the
 * worker pipeline. Workers turn knobs — they don't write new code.
 *
 * When idle, the scan_worker daemon runs systematic Great Lakes coverage
 * and the status is shown here.
 */

import React, { useState, useEffect, useRef, useCallback } from "react";
import { getApiBase } from "../services/api";
import ReportCard, { type SearchReport } from "./ReportCard";

// ── Types ─────────────────────────────────────────────────────────────────────

interface SearchJob {
  job_id: string;
  status: string;
  query: string;
  spec: {
    target_name: string;
    target_type: string;
    area_label: string;
    bbox: number[];
    date_range: string[];
    sensors: string[];
    passes: number[];
    knobs: Record<string, number>;
  };
  created: number;
  started: number | null;
  finished: number | null;
  error: string | null;
  result_summary: Record<string, unknown> | null;
}

interface IdleStatus {
  running?: boolean;
  status?: string;
  current_job?: string;
  jobs_completed?: number;
  message?: string;
  worker_id?: string;
}

// ── Example queries ───────────────────────────────────────────────────────────

const EXAMPLE_QUERIES = [
  "Find the Marquette and Bessemer No. 2 in Lake Erie",
  "Search for missing aircraft in Lake Michigan near Chicago",
  "Look for steel freighter wrecks in Lake Superior with strong magnetic signature",
  "Scan Straits of Mackinac for hydrocarbon leaks aggressively",
  "Find the Andaste in Lake Michigan south",
  "Search for missing person drift in Lake Huron",
  "Look for submerged vehicles near Lake Erie central basin",
  "Scan Lake Ontario for deep wrecks with thermal anomalies",
];

// ── Helpers ───────────────────────────────────────────────────────────────────

function elapsed(job: SearchJob): string {
  if (!job.started) return "";
  const end = job.finished ?? Date.now() / 1000;
  const s = Math.round(end - job.started);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

function statusColor(status: string): string {
  switch (status) {
    case "completed": return "#3fb950";
    case "running":   return "#58a6ff";
    case "queued":    return "#d29922";
    case "queued_to_worker": return "#d29922";
    case "failed":    return "#f85149";
    default:          return "#8b949e";
  }
}

function statusIcon(status: string): string {
  switch (status) {
    case "completed": return "✓";
    case "running":   return "⟳";
    case "queued":    return "⏳";
    case "queued_to_worker": return "⏳";
    case "failed":    return "✗";
    default:          return "·";
  }
}

function sensorBadge(sensor: string) {
  const colors: Record<string, string> = {
    thermal: "#ef4444", optical: "#3b82f6", sar: "#8b5cf6",
    magnetics: "#f59e0b", hls: "#10b981", swir: "#6366f1",
    ndvi: "#22c55e",
  };
  return (
    <span key={sensor} style={{
      display: "inline-block", padding: "1px 6px", borderRadius: 10,
      background: (colors[sensor] ?? "#444") + "33",
      border: `1px solid ${colors[sensor] ?? "#444"}`,
      color: colors[sensor] ?? "#aaa",
      fontSize: 10, marginRight: 3,
    }}>
      {sensor}
    </span>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function SearchPanel() {
  const [query, setQuery]           = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [jobs, setJobs]             = useState<SearchJob[]>([]);
  const [activeJob, setActiveJob]   = useState<SearchJob | null>(null);
  const [report, setReport]         = useState<SearchReport | null>(null);
  const [idleStatus, setIdleStatus] = useState<IdleStatus>({});
  const [error, setError]           = useState<string | null>(null);
  const [exampleIdx, setExampleIdx] = useState(0);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Rotate example placeholder
  useEffect(() => {
    const t = setInterval(() => setExampleIdx(i => (i + 1) % EXAMPLE_QUERIES.length), 4000);
    return () => clearInterval(t);
  }, []);

  // Load recent jobs on mount
  useEffect(() => {
    loadJobs();
    loadIdleStatus();
    const t = setInterval(loadIdleStatus, 15000);
    return () => clearInterval(t);
  }, []);

  const loadJobs = useCallback(async () => {
    try {
      const base = await getApiBase();
      const res = await fetch(`${base}/search?limit=10`);
      if (res.ok) {
        const data = await res.json();
        setJobs(data.jobs ?? []);
      }
    } catch { /* offline */ }
  }, []);

  const loadIdleStatus = useCallback(async () => {
    try {
      const base = await getApiBase();
      const res = await fetch(`${base}/search/idle/status`);
      if (res.ok) setIdleStatus(await res.json());
    } catch { /* offline */ }
  }, []);

  // Poll active job
  useEffect(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    if (!activeJob || activeJob.status === "completed" || activeJob.status === "failed") {
      if (activeJob?.status === "completed") loadReport(activeJob.job_id);
      return;
    }
    pollRef.current = setInterval(async () => {
      try {
        const base = await getApiBase();
        const res = await fetch(`${base}/search/${activeJob.job_id}`);
        if (res.ok) {
          const updated: SearchJob = await res.json();
          setActiveJob(updated);
          setJobs(prev => prev.map(j => j.job_id === updated.job_id ? updated : j));
          if (updated.status === "completed" || updated.status === "queued_to_worker") {
            clearInterval(pollRef.current!);
            loadReport(updated.job_id);
          } else if (updated.status === "failed") {
            clearInterval(pollRef.current!);
          }
        }
      } catch { /* transient */ }
    }, 3000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [activeJob?.job_id, activeJob?.status]);

  const loadReport = async (jobId: string) => {
    try {
      const base = await getApiBase();
      const res = await fetch(`${base}/search/${jobId}/report`);
      if (res.ok) setReport(await res.json());
    } catch { /* offline */ }
  };

  const handleSubmit = async () => {
    if (!query.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    setReport(null);
    try {
      const base = await getApiBase();
      const res = await fetch(`${base}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query.trim(), use_llm: true }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail ?? res.statusText);
      }
      const data = await res.json();
      const newJob: SearchJob = {
        job_id: data.job_id,
        status: data.status,
        query: query.trim(),
        spec: data.spec,
        created: Date.now() / 1000,
        started: null,
        finished: null,
        error: null,
        result_summary: null,
      };
      setActiveJob(newJob);
      setJobs(prev => [newJob, ...prev.slice(0, 9)]);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
    setSubmitting(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleSubmit();
  };

  const handleSelectJob = (job: SearchJob) => {
    setActiveJob(job);
    setReport(null);
    if (job.status === "completed" || job.status === "queued_to_worker") {
      loadReport(job.job_id);
    }
  };

  return (
    <div style={{
      display: "flex", flexDirection: "column", height: "100%",
      padding: "1.2rem 1.4rem", gap: "1rem", overflow: "hidden",
      background: "#0d1117", color: "#c9d1d9",
    }}>

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "baseline", gap: "0.8rem" }}>
        <h2 style={{ margin: 0, fontSize: "1.3rem", color: "#e6edf3" }}>
          🔍 Search & Rescue
        </h2>
        <span style={{ fontSize: 12, color: "#8b949e" }}>
          Type what you're looking for — the pipeline does the rest
        </span>
      </div>

      {/* ── Search box ─────────────────────────────────────────────────── */}
      <div style={{ display: "flex", gap: "0.6rem", alignItems: "flex-start" }}>
        <textarea
          ref={inputRef}
          rows={3}
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={EXAMPLE_QUERIES[exampleIdx]}
          style={{
            flex: 1, padding: "0.7rem 1rem", borderRadius: 8,
            border: "1px solid #30363d", background: "#161b22",
            color: "#e6edf3", fontSize: 14, fontFamily: "inherit",
            resize: "vertical", lineHeight: 1.5,
            outline: "none",
          }}
        />
        <button
          onClick={handleSubmit}
          disabled={submitting || !query.trim()}
          style={{
            padding: "0.7rem 1.4rem", borderRadius: 8, border: "none",
            background: submitting ? "#333" : "#1f6feb",
            color: "#fff", cursor: submitting ? "not-allowed" : "pointer",
            fontWeight: 700, fontSize: 14, whiteSpace: "nowrap",
            opacity: submitting ? 0.6 : 1,
          }}
        >
          {submitting ? "Queuing…" : "Search"}
        </button>
      </div>
      <div style={{ fontSize: 11, color: "#8b949e", marginTop: -8 }}>
        Ctrl+Enter to submit · Natural language accepted · LLM-refined when available
      </div>

      {error && (
        <div style={{ padding: "0.5rem 0.8rem", borderRadius: 6, background: "#f8514922", border: "1px solid #f85149", color: "#f85149", fontSize: 13 }}>
          {error}
        </div>
      )}

      {/* ── Main area: active job + history ────────────────────────────── */}
      <div style={{ display: "flex", gap: "1rem", flex: 1, overflow: "hidden", minHeight: 0 }}>

        {/* Left: active job + report */}
        <div style={{ flex: 2, display: "flex", flexDirection: "column", gap: "0.8rem", overflow: "auto" }}>

          {activeJob && (
            <div style={{
              border: "1px solid #30363d", borderRadius: 10,
              background: "#161b22", padding: "1rem",
            }}>
              {/* Job header */}
              <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "0.6rem" }}>
                <span style={{ fontSize: 18, color: statusColor(activeJob.status) }}>
                  {statusIcon(activeJob.status)}
                </span>
                <span style={{ fontWeight: 600, color: "#e6edf3", flex: 1 }}>
                  {activeJob.query}
                </span>
                <span style={{ fontSize: 11, color: statusColor(activeJob.status), fontWeight: 600 }}>
                  {activeJob.status.replace("_", " ").toUpperCase()}
                </span>
                {activeJob.started && (
                  <span style={{ fontSize: 11, color: "#8b949e" }}>{elapsed(activeJob)}</span>
                )}
              </div>

              {/* Parsed spec */}
              {activeJob.spec && (
                <div style={{
                  display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.4rem 1rem",
                  fontSize: 12, color: "#8b949e", marginBottom: "0.6rem",
                }}>
                  <div><span style={{ color: "#58a6ff" }}>Target:</span> {activeJob.spec.target_name}</div>
                  <div><span style={{ color: "#58a6ff" }}>Area:</span> {activeJob.spec.area_label}</div>
                  <div><span style={{ color: "#58a6ff" }}>Dates:</span> {activeJob.spec.date_range?.join(" → ")}</div>
                  <div>
                    <span style={{ color: "#58a6ff" }}>Sensors:</span>{" "}
                    {activeJob.spec.sensors?.map(sensorBadge)}
                  </div>
                  <div>
                    <span style={{ color: "#58a6ff" }}>Passes:</span>{" "}
                    {activeJob.spec.passes?.map(p => (
                      <span key={p} style={{ marginRight: 4, color: "#d29922" }}>#{p}</span>
                    ))}
                  </div>
                  <div>
                    <span style={{ color: "#58a6ff" }}>Sensitivity:</span>{" "}
                    {activeJob.spec.knobs?.hc_threshold?.toFixed(1) ?? "2.0"}
                  </div>
                </div>
              )}

              {activeJob.status === "running" && (
                <div style={{ fontSize: 12, color: "#58a6ff", display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>⟳</span>
                  Running pipeline… workers are scanning
                </div>
              )}

              {activeJob.error && (
                <div style={{ fontSize: 12, color: "#f85149", marginTop: 4 }}>
                  Error: {activeJob.error}
                </div>
              )}
            </div>
          )}

          {/* Report */}
          {report !== null && <ReportCard report={report} />}

          {/* Empty state */}
          {!activeJob && !report && (
            <div style={{
              flex: 1, display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center",
              color: "#8b949e", gap: "0.8rem", padding: "2rem",
            }}>
              <div style={{ fontSize: 48 }}>🌊</div>
              <div style={{ fontSize: 14, textAlign: "center", maxWidth: 400 }}>
                Type what you're looking for above. The pipeline will parse your
                request, tune the sensor knobs, and run the scan automatically.
              </div>
              <div style={{ fontSize: 12, color: "#6e7681", textAlign: "center" }}>
                Examples: wreck names, aircraft, missing persons, hydrocarbon leaks,
                specific lakes, date ranges, sensitivity levels
              </div>
            </div>
          )}
        </div>

        {/* Right: history + idle status */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "0.8rem", overflow: "auto", minWidth: 220 }}>

          {/* Idle worker status */}
          <div style={{
            border: "1px solid #30363d", borderRadius: 10,
            background: "#161b22", padding: "0.8rem",
          }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: "#8b949e", marginBottom: "0.5rem" }}>
              💤 Idle Worker
            </div>
            {idleStatus.running ? (
              <div>
                <div style={{ fontSize: 12, color: "#3fb950", marginBottom: 2 }}>● Running</div>
                {idleStatus.current_job && (
                  <div style={{ fontSize: 11, color: "#8b949e" }}>
                    Job: {String(idleStatus.current_job).slice(0, 40)}
                  </div>
                )}
                {idleStatus.jobs_completed !== undefined && (
                  <div style={{ fontSize: 11, color: "#8b949e" }}>
                    Completed: {idleStatus.jobs_completed}
                  </div>
                )}
              </div>
            ) : (
              <div style={{ fontSize: 12, color: "#6e7681" }}>
                {idleStatus.message ?? "Not running — start scan_worker.py"}
              </div>
            )}
            {idleStatus.worker_id && (
              <div style={{ fontSize: 10, color: "#6e7681", marginTop: 4 }}>
                Node: {idleStatus.worker_id}
              </div>
            )}
          </div>

          {/* Recent searches */}
          {jobs.length > 0 && (
            <div style={{
              border: "1px solid #30363d", borderRadius: 10,
              background: "#161b22", padding: "0.8rem", flex: 1, overflow: "auto",
            }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#8b949e", marginBottom: "0.5rem" }}>
                Recent Searches
              </div>
              {jobs.map(job => (
                <div
                  key={job.job_id}
                  onClick={() => handleSelectJob(job)}
                  style={{
                    padding: "0.5rem 0.4rem", borderBottom: "1px solid #21262d",
                    cursor: "pointer", borderRadius: 4,
                    background: activeJob?.job_id === job.job_id ? "#1f2937" : "transparent",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ color: statusColor(job.status), fontSize: 12 }}>
                      {statusIcon(job.status)}
                    </span>
                    <span style={{ fontSize: 12, color: "#e6edf3", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {job.query}
                    </span>
                  </div>
                  <div style={{ fontSize: 10, color: "#6e7681", marginTop: 2, paddingLeft: 18 }}>
                    {job.spec?.area_label} · {job.spec?.target_type} · {elapsed(job)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
