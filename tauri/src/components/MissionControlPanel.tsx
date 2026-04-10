/**
 * CESAROPS Mission Control Panel
 * ─────────────────────────────────────────────────────────────────
 * Unified scan launcher.  The agent (or the user) picks a preset or
 * builds a custom mission — bbox, date range, which passes to run,
 * and the per-band intensity thresholds ("band mixing").
 *
 * Communicates with:
 *   Tauri command: run_mission(mission_json, work_dir)  →  TaskOutput
 *   Python:        cesarops_mission.py  --mission-json
 */

import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

// ── Pass meta ─────────────────────────────────────────────────────────────────

interface PassMeta {
  label: string;
  icon: string;
  desc: string;
  color: string;
  sliders: SliderDef[];
}

interface SliderDef {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  defaultVal: number;
  /** If true the UI shows the negative/positive direction labels reversed */
  invertedLabel?: boolean;
}

const PASS_META: Record<string, PassMeta> = {
  standard: {
    label: "Standard Anomaly",
    icon: "🔬",
    color: "#4a90d9",
    desc: "Z-score optical / SAR / thermal pixel anomaly. Runs on all bands in the data directory.",
    sliders: [
      { key: "threshold", label: "Sigma cutoff", min: 0.5, max: 4.0, step: 0.1, defaultVal: 1.5 },
    ],
  },
  hydrocarbon: {
    label: "Hydrocarbon HC",
    icon: "🛢",
    color: "#e07b20",
    desc: "B11 SWIR dark anomaly + B04 Red bright confirm — oil sheen / fuel slick detection.",
    sliders: [
      { key: "swir_thresh", label: "SWIR B11 σ (negative = darker)", min: -0.5, max: -3.5, step: -0.1, defaultVal: -1.8 },
      { key: "red_thresh",  label: "Red B04 σ (positive = brighter)",  min: 0.5, max: 3.0, step: 0.1, defaultVal: 1.5 },
    ],
  },
  thermal: {
    label: "Thermal Cold-Sink",
    icon: "🌡",
    color: "#56c5d0",
    desc: "B10 LWIR cold-sink mode — submerged steel hull lies cooler than surrounding sediment.",
    sliders: [
      { key: "threshold", label: "Cold-sink σ", min: 0.5, max: 4.0, step: 0.1, defaultVal: 2.0 },
    ],
  },
  stumpf: {
    label: "Stumpf Bathymetric",
    icon: "📐",
    color: "#78c96d",
    desc: "Log-ratio B02/B03 shallow-bottom anomaly — detects hard structures in clear shallow water.",
    sliders: [
      { key: "threshold", label: "Shallow σ", min: 0.5, max: 4.0, step: 0.1, defaultVal: 2.0 },
    ],
  },
  nauticuvs: {
    label: "NauticUVs LoG",
    icon: "🔵",
    color: "#9778cc",
    desc: "Multi-scale Laplacian-of-Gaussian blob on B02 + B10 — hull curvature energy proxy.",
    sliders: [
      { key: "energy_threshold", label: "LoG energy σ", min: 1.0, max: 6.0, step: 0.1, defaultVal: 3.5 },
      { key: "top_n", label: "Top-N results", min: 10, max: 200, step: 10, defaultVal: 50 },
    ],
  },
  swir_silt_erasure: {
    label: "SWIR Silt Erasure",
    icon: "🪨",
    color: "#c68a3b",
    desc: "B11/B12 ratio anomaly — ferrous metal beneath silt appears as elevated SWIR ratio (Marquette-Bessemer specialist).",
    sliders: [
      { key: "threshold", label: "B11/B12 ratio σ", min: 0.5, max: 4.0, step: 0.1, defaultVal: 2.5 },
      { key: "top_n", label: "Top-N results", min: 10, max: 100, step: 10, defaultVal: 30 },
    ],
  },
  mussel_clearspot: {
    label: "Mussel Clear-Spot",
    icon: "🐚",
    color: "#d4a84b",
    desc: "Positive B02 bias in turbid Erie background — Dreissenid mussel colony filter clears silt and exposes hull.",
    sliders: [
      { key: "threshold", label: "B02 elevated σ", min: 0.5, max: 4.0, step: 0.1, defaultVal: 2.0 },
      { key: "top_n", label: "Top-N results", min: 10, max: 100, step: 10, defaultVal: 30 },
    ],
  },
};

// ── Mission presets ───────────────────────────────────────────────────────────

interface MissionPreset {
  label: string;
  bbox: [number, number, number, number]; // [lat_min, lon_min, lat_max, lon_max]
  data_dirs: string[];
  passes: Record<string, Record<string, unknown>>;
  output_tag: string;
}

const PRESETS: Record<string, MissionPreset> = {
  triple_lock_erie: {
    label: "🔒 Triple Lock — Lake Erie",
    bbox: [41.30, -83.50, 42.50, -78.80],
    output_tag: "triple_lock_erie",
    data_dirs: ["downloads/erie", "downloads/hls"],
    passes: {
      standard:          { enabled: true,  threshold: 1.5 },
      hydrocarbon:       { enabled: true,  swir_thresh: -1.8, red_thresh: 1.5 },
      thermal:           { enabled: true,  threshold: 2.0 },
      stumpf:            { enabled: false, threshold: 2.0 },
      nauticuvs:         { enabled: true,  energy_threshold: 3.5, top_n: 50 },
      swir_silt_erasure: { enabled: false, threshold: 2.5, top_n: 30 },
      mussel_clearspot:  { enabled: false, threshold: 2.0, top_n: 30 },
    },
  },
  mb2_wreck_hunt: {
    label: "⚓ Marquette & Bessemer No. 2",
    bbox: [41.80, -82.50, 42.50, -80.00],
    output_tag: "mb2_hunt",
    data_dirs: ["downloads/erie", "downloads/hls"],
    passes: {
      standard:          { enabled: true,  threshold: 1.5 },
      hydrocarbon:       { enabled: true,  swir_thresh: -1.8, red_thresh: 1.5 },
      thermal:           { enabled: true,  threshold: 1.8 },
      stumpf:            { enabled: true,  threshold: 2.0 },
      nauticuvs:         { enabled: true,  energy_threshold: 3.0, top_n: 50 },
      swir_silt_erasure: { enabled: true,  threshold: 2.5, top_n: 30 },
      mussel_clearspot:  { enabled: true,  threshold: 2.0, top_n: 30 },
    },
  },
  erie_hc_timeline: {
    label: "🌊 Erie HC Timeline (oil/fuel)",
    bbox: [41.30, -83.50, 42.50, -78.80],
    output_tag: "erie_hc_timeline",
    data_dirs: ["downloads/erie", "downloads/hls"],
    passes: {
      standard:          { enabled: true,  threshold: 1.5 },
      hydrocarbon:       { enabled: true,  swir_thresh: -1.8, red_thresh: 1.5 },
      thermal:           { enabled: false, threshold: 2.0 },
      stumpf:            { enabled: false, threshold: 2.0 },
      nauticuvs:         { enabled: false, energy_threshold: 3.5, top_n: 50 },
      swir_silt_erasure: { enabled: false, threshold: 2.5, top_n: 30 },
      mussel_clearspot:  { enabled: false, threshold: 2.0, top_n: 30 },
    },
  },
  straits_triple_lock: {
    label: "🌉 Triple Lock — Straits of Mackinac",
    bbox: [45.70, -84.90, 46.05, -84.10],
    output_tag: "straits_triple_lock",
    data_dirs: ["downloads/straits", "downloads/michigan", "downloads/hls"],
    passes: {
      standard:          { enabled: true,  threshold: 1.5 },
      hydrocarbon:       { enabled: true,  swir_thresh: -1.8, red_thresh: 1.5 },
      thermal:           { enabled: true,  threshold: 2.0 },
      stumpf:            { enabled: true,  threshold: 2.0 },
      nauticuvs:         { enabled: true,  energy_threshold: 3.5, top_n: 50 },
      swir_silt_erasure: { enabled: false, threshold: 2.5, top_n: 30 },
      mussel_clearspot:  { enabled: false, threshold: 2.0, top_n: 30 },
    },
  },
  andaste_hunt: {
    label: "👻 Andaste — Lake Michigan South",
    bbox: [42.30, -88.50, 43.20, -87.40],
    output_tag: "andaste_hunt",
    data_dirs: ["downloads/michigan", "downloads/hls"],
    passes: {
      standard:          { enabled: true,  threshold: 1.2 },
      hydrocarbon:       { enabled: true,  swir_thresh: -1.8, red_thresh: 1.5 },
      thermal:           { enabled: true,  threshold: 1.8 },
      stumpf:            { enabled: true,  threshold: 2.0 },
      nauticuvs:         { enabled: true,  energy_threshold: 3.0, top_n: 50 },
      swir_silt_erasure: { enabled: false, threshold: 2.5, top_n: 30 },
      mussel_clearspot:  { enabled: false, threshold: 2.0, top_n: 30 },
    },
  },
};

// ── Default / blank state ─────────────────────────────────────────────────────

function defaultPassValues(): Record<string, Record<string, unknown>> {
  const out: Record<string, Record<string, unknown>> = {};
  for (const [key, meta] of Object.entries(PASS_META)) {
    out[key] = { enabled: false };
    meta.sliders.forEach((s) => {
      out[key][s.key] = s.defaultVal;
    });
  }
  return out;
}

function applyPreset(preset: MissionPreset): Record<string, Record<string, unknown>> {
  const base = defaultPassValues();
  for (const [key, cfg] of Object.entries(preset.passes)) {
    if (base[key]) {
      base[key] = { ...base[key], ...cfg };
    }
  }
  return base;
}

// ── Quick-bbox buttons ────────────────────────────────────────────────────────

const QUICK_BBOXES: Array<{ label: string; bbox: [number, number, number, number] }> = [
  { label: "Lake Erie",          bbox: [41.30, -83.50, 42.50, -78.80] },
  { label: "Erie Central Basin", bbox: [41.80, -82.50, 42.50, -80.00] },
  { label: "Lake Michigan",      bbox: [41.60, -87.80, 46.10, -84.70] },
  { label: "Michigan South",     bbox: [42.30, -88.50, 43.20, -87.40] },
  { label: "Straits Mackinac",   bbox: [45.70, -84.90, 46.05, -84.10] },
  { label: "Lake Huron",         bbox: [43.00, -84.00, 46.50, -79.50] },
  { label: "Lake Superior",      bbox: [46.40, -92.00, 48.20, -84.40] },
  { label: "Lake Ontario",       bbox: [43.20, -79.90, 44.20, -76.00] },
];

// ── Component ─────────────────────────────────────────────────────────────────

export default function MissionControlPanel() {
  const [missionName, setMissionName]     = useState("Custom Mission");
  const [outputTag,   setOutputTag]       = useState("custom_mission");
  const [selectedPreset, setSelectedPreset] = useState("custom");
  const [bbox, setBbox] = useState<[number, number, number, number]>([41.30, -83.50, 42.50, -78.80]);
  const [dataDirs, setDataDirs]           = useState("downloads/erie,downloads/hls");
  const [passes, setPasses]               = useState<Record<string, Record<string, unknown>>>(defaultPassValues);
  const [subZones, setSubZones]           = useState<Array<{ name: string; bbox_str: string; passes_str: string }>>([]);

  const [output,  setOutput]  = useState("");
  const [running, setRunning] = useState(false);
  const [workDir, setWorkDir] = useState("");
  const [result,  setResult]  = useState<Record<string, unknown> | null>(null);
  const [showSubZones, setShowSubZones] = useState(false);

  const outputRef = useRef<HTMLTextAreaElement>(null);

  // Resolve work dir from Rust
  useEffect(() => {
    invoke<string>("get_work_dir")
      .then((d) => setWorkDir(d))
      .catch(() => setWorkDir("../"));
  }, []);

  // Auto-scroll output
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [output]);

  // Apply a built-in preset
  const applyBuiltinPreset = (key: string) => {
    setSelectedPreset(key);
    if (key === "custom") return;
    const p = PRESETS[key];
    if (!p) return;
    setMissionName(p.label.replace(/^[^ ]+ /, ""));
    setOutputTag(p.output_tag);
    setBbox(p.bbox);
    setDataDirs(p.data_dirs.join(","));
    setPasses(applyPreset(p));
  };

  // Toggle a pass on/off
  const togglePass = (passKey: string) => {
    setPasses((prev) => ({
      ...prev,
      [passKey]: { ...prev[passKey], enabled: !prev[passKey].enabled },
    }));
  };

  // Update a slider value
  const setSlider = (passKey: string, sliderKey: string, val: number) => {
    setPasses((prev) => ({
      ...prev,
      [passKey]: { ...prev[passKey], [sliderKey]: val },
    }));
  };

  // Build the mission JSON to send to Python
  const buildMissionJson = (): string => {
    const sz = subZones
      .filter((z) => z.name.trim())
      .map((z) => {
        const parts = z.bbox_str.split(",").map((x) => parseFloat(x.trim()));
        return {
          name: z.name,
          bbox: parts.length === 4 ? parts : bbox,
          passes: z.passes_str
            .split(",")
            .map((x) => x.trim())
            .filter(Boolean),
        };
      });

    return JSON.stringify({
      name:       missionName,
      bbox:       Array.from(bbox),
      output_tag: outputTag || "mission",
      data_dirs:  dataDirs.split(",").map((d) => d.trim()).filter(Boolean),
      passes,
      sub_zones:  sz,
    });
  };

  const handleLaunch = async () => {
    setRunning(true);
    setOutput("");
    setResult(null);

    const missionJson = buildMissionJson();
    const wd = workDir || "../";

    // Listen for streaming task_output events
    const unlisten = await listen<{ task_id: string; line: string; is_error: boolean }>(
      "task_output",
      (event) => {
        const { line } = event.payload;
        setOutput((prev) => prev + line + "\n");

        // Parse structured result line
        if (line.includes("[MISSION_RESULT]")) {
          try {
            const jsonStr = line.split("[MISSION_RESULT]")[1].trim();
            const parsed = JSON.parse(jsonStr);
            setResult(parsed);
          } catch (_) { /* non-blocking */ }
        }
      }
    );

    try {
      const res = await invoke<{ stdout: string; stderr: string; status: string }>(
        "run_mission",
        { missionJson, workDir: wd }
      );
      if (res.stdout) setOutput((prev) => prev + res.stdout);
      if (res.stderr) setOutput((prev) => prev + "\n--- stderr ---\n" + res.stderr);
    } catch (e: unknown) {
      setOutput((prev) => prev + `\n[ERROR] ${String(e)}`);
    } finally {
      unlisten();
      setRunning(false);
    }
  };

  const enabledPasses = Object.entries(passes).filter(([, v]) => v.enabled).map(([k]) => k);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", padding: "0 16px 16px", overflow: "auto", fontFamily: "inherit" }}>

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div style={{ borderBottom: "1px solid #2a3a4a", paddingBottom: 12, marginBottom: 14 }}>
        <h2 style={{ margin: 0, fontSize: 18, color: "#7ec8e3", letterSpacing: 1 }}>
          🎯 MISSION CONTROL
        </h2>
        <p style={{ margin: "4px 0 0", fontSize: 12, color: "#7a8fa0" }}>
          Configure multi-pass scan missions. Toggle passes on/off and tune band intensities for best results.
        </p>
      </div>

      {/* ── Preset selector ──────────────────────────────────────────────── */}
      <table style={{ width: "100%", marginBottom: 14, borderCollapse: "collapse" }}>
        <tbody>
          <tr>
            <td style={{ width: 110, color: "#7a8fa0", fontSize: 12, paddingBottom: 6 }}>Preset</td>
            <td>
              <select
                value={selectedPreset}
                onChange={(e) => applyBuiltinPreset(e.target.value)}
                style={selectStyle}
              >
                <option value="custom">— Custom —</option>
                {Object.entries(PRESETS).map(([k, p]) => (
                  <option key={k} value={k}>{p.label}</option>
                ))}
              </select>
            </td>
          </tr>
          <tr>
            <td style={{ color: "#7a8fa0", fontSize: 12, paddingBottom: 6 }}>Mission name</td>
            <td>
              <input
                value={missionName}
                onChange={(e) => setMissionName(e.target.value)}
                style={inputStyle}
                placeholder="Triple Lock Lake Erie"
              />
            </td>
          </tr>
          <tr>
            <td style={{ color: "#7a8fa0", fontSize: 12 }}>Output tag</td>
            <td>
              <input
                value={outputTag}
                onChange={(e) => setOutputTag(e.target.value)}
                style={inputStyle}
                placeholder="my_mission_oct2015"
              />
            </td>
          </tr>
        </tbody>
      </table>

      {/* ── Bbox ─────────────────────────────────────────────────────────── */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ color: "#7a8fa0", fontSize: 12, marginBottom: 6 }}>
          Bounding Box&nbsp;<span style={{ color: "#4a6a80" }}>(lat_min, lon_min, lat_max, lon_max)</span>
        </div>
        <div style={{ display: "flex", gap: 6, marginBottom: 8, flexWrap: "wrap" }}>
          {(["lat_min", "lon_min", "lat_max", "lon_max"] as const).map((field, i) => (
            <div key={field} style={{ display: "flex", flexDirection: "column", minWidth: 90 }}>
              <span style={{ fontSize: 10, color: "#4a6a80", marginBottom: 2 }}>{field}</span>
              <input
                type="number"
                step="0.01"
                value={bbox[i]}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (!isNaN(v)) {
                    setBbox((b) => {
                      const nb: [number, number, number, number] = [...b] as [number, number, number, number];
                      nb[i] = v;
                      return nb;
                    });
                  }
                }}
                style={{ ...inputStyle, width: 90, padding: "4px 6px" }}
              />
            </div>
          ))}
        </div>
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {QUICK_BBOXES.map((qb) => (
            <button
              key={qb.label}
              onClick={() => setBbox(qb.bbox)}
              style={chipButtonStyle}
            >
              {qb.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── Data dirs ────────────────────────────────────────────────────── */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ color: "#7a8fa0", fontSize: 12, marginBottom: 4 }}>
          Data directories&nbsp;<span style={{ color: "#4a6a80" }}>(comma-separated, relative to cesarops-core/)</span>
        </div>
        <input
          value={dataDirs}
          onChange={(e) => setDataDirs(e.target.value)}
          style={{ ...inputStyle, width: "100%" }}
          placeholder="downloads/erie,downloads/hls"
        />
      </div>

      {/* ── Pass cards ───────────────────────────────────────────────────── */}
      <div style={{ color: "#7a8fa0", fontSize: 12, marginBottom: 8 }}>
        Detection Passes
        <span style={{ marginLeft: 8, color: "#3a7a3a", fontSize: 11 }}>
          {enabledPasses.length} active
        </span>
      </div>
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
        gap: 10,
        marginBottom: 14,
      }}>
        {Object.entries(PASS_META).map(([key, meta]) => {
          const passCfg = passes[key] || {};
          const active = !!passCfg.enabled;
          return (
            <div key={key} style={{
              background: active ? "#0e1e2e" : "#0a1520",
              border: `1px solid ${active ? meta.color : "#1a2a3a"}`,
              borderRadius: 8,
              padding: "10px 12px",
              transition: "border-color 0.2s",
              opacity: active ? 1 : 0.6,
            }}>
              {/* Card header */}
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <span style={{ fontSize: 18 }}>{meta.icon}</span>
                <span style={{ color: active ? meta.color : "#4a6a80", fontWeight: 600, fontSize: 13, flex: 1 }}>
                  {meta.label}
                </span>
                {/* Toggle switch */}
                <div
                  onClick={() => togglePass(key)}
                  style={{
                    width: 36, height: 20, borderRadius: 10, cursor: "pointer",
                    background: active ? meta.color : "#1a2a3a",
                    position: "relative", transition: "background 0.2s", flexShrink: 0,
                  }}
                >
                  <div style={{
                    position: "absolute",
                    top: 3, left: active ? 18 : 3,
                    width: 14, height: 14, borderRadius: "50%",
                    background: "#fff", transition: "left 0.2s",
                  }} />
                </div>
              </div>
              {/* Description */}
              <p style={{ margin: "0 0 8px", fontSize: 11, color: "#5a7a8a", lineHeight: 1.4 }}>
                {meta.desc}
              </p>
              {/* Intensity sliders */}
              {active && meta.sliders.map((sl) => {
                const raw = passCfg[sl.key] as number ?? sl.defaultVal;
                const isNegRange = sl.max < sl.min; // inverted range (SWIR thresh)
                const lo = isNegRange ? sl.max : sl.min;
                const hi = isNegRange ? sl.min : sl.max;
                return (
                  <div key={sl.key} style={{ marginBottom: 6 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#7a8fa0" }}>
                      <span>{sl.label}</span>
                      <span style={{ color: meta.color, fontWeight: 600 }}>{raw.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min={lo}
                      max={hi}
                      step={Math.abs(sl.step)}
                      value={Math.abs(raw)}
                      onChange={(e) => {
                        let v = parseFloat(e.target.value);
                        if (isNegRange) v = -v;
                        setSlider(key, sl.key, v);
                      }}
                      style={{ width: "100%", accentColor: meta.color, cursor: "pointer" }}
                    />
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>

      {/* ── Sub-zones ────────────────────────────────────────────────────── */}
      <div style={{ marginBottom: 14 }}>
        <button
          onClick={() => setShowSubZones((v) => !v)}
          style={chipButtonStyle}
        >
          {showSubZones ? "▼" : "▶"} Sub-zones ({subZones.length})
        </button>
        {showSubZones && (
          <div style={{ marginTop: 8, padding: "10px 12px", background: "#0a1520", border: "1px solid #1a2a3a", borderRadius: 8 }}>
            {subZones.map((sz, i) => (
              <div key={i} style={{ display: "flex", gap: 6, marginBottom: 8, flexWrap: "wrap", alignItems: "flex-end" }}>
                <div style={{ display: "flex", flexDirection: "column" }}>
                  <span style={{ fontSize: 10, color: "#4a6a80" }}>Zone name</span>
                  <input
                    value={sz.name}
                    onChange={(e) => setSubZones((z) => z.map((x, j) => j === i ? { ...x, name: e.target.value } : x))}
                    style={{ ...inputStyle, width: 140 }}
                    placeholder="M&B2 Central Basin"
                  />
                </div>
                <div style={{ display: "flex", flexDirection: "column" }}>
                  <span style={{ fontSize: 10, color: "#4a6a80" }}>bbox (latMin,lonMin,latMax,lonMax)</span>
                  <input
                    value={sz.bbox_str}
                    onChange={(e) => setSubZones((z) => z.map((x, j) => j === i ? { ...x, bbox_str: e.target.value } : x))}
                    style={{ ...inputStyle, width: 220 }}
                    placeholder="41.8,-82.5,42.5,-80.0"
                  />
                </div>
                <div style={{ display: "flex", flexDirection: "column" }}>
                  <span style={{ fontSize: 10, color: "#4a6a80" }}>passes (comma-sep)</span>
                  <input
                    value={sz.passes_str}
                    onChange={(e) => setSubZones((z) => z.map((x, j) => j === i ? { ...x, passes_str: e.target.value } : x))}
                    style={{ ...inputStyle, width: 200 }}
                    placeholder="swir_silt_erasure,mussel_clearspot"
                  />
                </div>
                <button
                  onClick={() => setSubZones((z) => z.filter((_, j) => j !== i))}
                  style={{ ...chipButtonStyle, background: "#3a1010", border: "1px solid #7a2020", color: "#cc7070" }}
                >✕ Remove</button>
              </div>
            ))}
            <button
              onClick={() => setSubZones((z) => [...z, { name: "", bbox_str: bbox.join(","), passes_str: "" }])}
              style={chipButtonStyle}
            >+ Add sub-zone</button>
          </div>
        )}
      </div>

      {/* ── Mission summary badge strip ─────────────────────────────────── */}
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 14 }}>
        <span style={badgeStyle("#1a3a2a", "#3a7a3a")}>
          Bbox: {bbox[0]}°N {bbox[1]}°E → {bbox[2]}°N {bbox[3]}°E
        </span>
        {enabledPasses.map((p) => (
          <span key={p} style={badgeStyle("#1a2a3a", PASS_META[p]?.color ?? "#4a90d9")}>
            {PASS_META[p]?.icon} {PASS_META[p]?.label ?? p}
          </span>
        ))}
        {enabledPasses.length === 0 && (
          <span style={badgeStyle("#3a1a1a", "#cc4444")}>⚠ No passes enabled</span>
        )}
      </div>

      {/* ── Launch button ────────────────────────────────────────────────── */}
      <button
        onClick={handleLaunch}
        disabled={running || enabledPasses.length === 0}
        style={{
          padding: "10px 24px",
          background: running ? "#1a2a1a" : "#1a4a2a",
          border: `1px solid ${running ? "#2a4a2a" : "#3a8a4a"}`,
          borderRadius: 6,
          color: running ? "#5a8a5a" : "#7afa7a",
          fontWeight: 700,
          fontSize: 14,
          cursor: running || enabledPasses.length === 0 ? "not-allowed" : "pointer",
          marginBottom: 14,
          letterSpacing: 1,
        }}
      >
        {running ? "⏳ RUNNING MISSION…" : "🚀 LAUNCH MISSION"}
      </button>

      {/* ── Output stream ────────────────────────────────────────────────── */}
      {(output || running) && (
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 12, color: "#7a8fa0", marginBottom: 4 }}>Mission output</div>
          <textarea
            ref={outputRef}
            readOnly
            value={output}
            style={{
              width: "100%",
              height: 260,
              background: "#040d14",
              color: "#7ae07a",
              border: "1px solid #1a2a3a",
              borderRadius: 6,
              padding: "8px 10px",
              fontFamily: "monospace",
              fontSize: 11,
              resize: "vertical",
              boxSizing: "border-box",
            }}
          />
        </div>
      )}

      {/* ── Result summary ───────────────────────────────────────────────── */}
      {result && (
        <div style={{
          padding: "12px 16px",
          background: "#0a1e0a",
          border: "1px solid #2a6a2a",
          borderRadius: 8,
          marginBottom: 14,
        }}>
          <div style={{ fontWeight: 700, color: "#7afa7a", marginBottom: 8, fontSize: 14 }}>
            ✅ Mission Complete — {String(result.name)}
          </div>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <Stat label="Total detections" value={String(result.detections ?? 0)} color="#7afa7a" />
            <Stat label="Hydrocarbon"      value={String(result.hc_count ?? 0)}    color="#f0a050" />
            <Stat label="Known wreck hits" value={String(result.known_hits ?? 0)}  color="#50c8f0" />
          </div>
          {!!result.output_dir && (
            <div style={{ marginTop: 8, fontSize: 11, color: "#4a9a4a" }}>
              Output: {String(result.output_dir)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Tiny helpers ──────────────────────────────────────────────────────────────

function Stat({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      <span style={{ fontSize: 10, color: "#4a7a4a" }}>{label}</span>
      <span style={{ fontSize: 22, fontWeight: 700, color }}>{value}</span>
    </div>
  );
}

function badgeStyle(bg: string, border: string): React.CSSProperties {
  return {
    background: bg,
    border: `1px solid ${border}`,
    borderRadius: 4,
    padding: "2px 8px",
    fontSize: 11,
    color: border,
    whiteSpace: "nowrap",
  };
}

const selectStyle: React.CSSProperties = {
  background: "#0a1520",
  border: "1px solid #1e3a4a",
  borderRadius: 4,
  color: "#a0c8d8",
  padding: "4px 8px",
  fontSize: 13,
  width: "100%",
};

const inputStyle: React.CSSProperties = {
  background: "#0a1520",
  border: "1px solid #1e3a4a",
  borderRadius: 4,
  color: "#c0d8e8",
  padding: "4px 8px",
  fontSize: 13,
};

const chipButtonStyle: React.CSSProperties = {
  background: "#0e1e2e",
  border: "1px solid #2a4a5a",
  borderRadius: 4,
  color: "#5a8aa0",
  padding: "3px 10px",
  fontSize: 11,
  cursor: "pointer",
};
