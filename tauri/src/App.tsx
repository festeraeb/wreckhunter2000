import { useEffect, useState } from "react";
import type { Wreck, ActivePanel, MagPotential } from "./types";
import WreckList from "./components/WreckList";
import WreckDetail from "./components/WreckDetail";
import { PasswordGate } from "./components/PasswordGate";

import StatsPanel from "./components/StatsPanel";
import ScanPanel from "./components/ScanPanel";
import MagPipelinePanel from "./components/MagPipelinePanel";
import EriePanel from "./components/EriePanel";
import RestorationPanel from "./components/RestorationPanel";
import MapPanel from "./components/MapPanel";
import { EarthMonitorPanel } from "./components/EarthMonitorPanel";
import PDFBreakerPanel from "./components/PDFBreakerPanel";
import ExportPanel from "./components/ExportPanel";
import AgentPanel from "./components/AgentPanel";
import MissionControlPanel from "./components/MissionControlPanel";
import LoranPanel from "./components/LoranPanel";
import ExtendedSensorsPanel from "./components/ExtendedSensorsPanel";
import HarvesterPanel from "./components/HarvesterPanel";
import SatelliteTrackerPanel from "./components/SatelliteTrackerPanel";
import AutoBagPrompt from "./components/AutoBagPrompt";
import UpdateChecker from "./components/UpdateChecker";
import WorkerStatusPanel from "./components/WorkerStatusPanel";
import IdleControlPanel from "./components/IdleControlPanel";
import ScanOutputPanel from "./components/ScanOutputPanel";
import HardwareDashboard from "./components/HardwareDashboard";
import KoboldAgentPanel from "./components/KoboldAgentPanel";
import SearchPanel from "./components/SearchPanel";
import { getApiBase, resetConnectionState } from "./services/api";
import "./styles/global.css";

// Panels accessible without authentication
const PUBLIC_PANELS: ActivePanel[] = ["stats", "list", "map", "satellite", "detail", "scan-output", "search"];

export default function App() {
  // Restore session immediately — don't wait for a locked panel to be clicked
  const [unlocked, setUnlocked] = useState(() => sessionStorage.getItem("wh2k_unlocked") === "1");
  // Show gate on startup if not already unlocked via session
  const [showGate, setShowGate] = useState(() => sessionStorage.getItem("wh2k_unlocked") !== "1");
  const [panel, setPanel] = useState<ActivePanel>("search");
  const [selectedWreck, setSelectedWreck] = useState<Wreck | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [online, setOnline] = useState<boolean | null>(null);
  const [apiBase, setApiBase] = useState<string>("resolving...");

  // Quick-filter state — passed as initial values to WreckList
  const [quickSteelOnly, setQuickSteelOnly] = useState(false);
  const [quickMagFilter, setQuickMagFilter] = useState<MagPotential | "">("");
  // listKey increments to force WreckList remount when quick filters change
  const [listKey, setListKey] = useState(0);

  // Check backend connectivity on mount
  useEffect(() => {
    getApiBase()
      .then((base) => {
        setApiBase(base);
        return fetch(`${base}/health`, { signal: AbortSignal.timeout(2000) });
      })
      .then(() => setOnline(true))
      .catch(() => setOnline(false));
  }, []);

  const handleSelect = (w: Wreck) => {
    setSelectedWreck(w);
    setPanel("detail");
  };

  const handleRetryConnection = () => {
    resetConnectionState();
    getApiBase()
      .then((base) => {
        setApiBase(base);
        return fetch(`${base}/health`, { signal: AbortSignal.timeout(2000) });
      })
      .then(() => setOnline(true))
      .catch(() => setOnline(false));
  };

  /** Navigate to wreck list with preset filters. */
  const goListWithFilter = (steelOnly: boolean, magFilter: MagPotential | "") => {
    setQuickSteelOnly(steelOnly);
    setQuickMagFilter(magFilter);
    setListKey(k => k + 1); // remount WreckList so it picks up new initial values
    setPanel("list");
  };

  /** Navigate to a panel — prompt for password if it's protected and not yet unlocked */
  const handleSetPanel = (target: ActivePanel) => {
    if (!unlocked && !PUBLIC_PANELS.includes(target)) {
      setShowGate(true);
      return;
    }
    setPanel(target);
  };

  return (
    <div className="app-shell">
      {showGate && (
        <PasswordGate onUnlock={() => { setUnlocked(true); setShowGate(false); }} />
      )}
      <AutoBagPrompt />
      <UpdateChecker />
      {/* ── Top bar ────────────────────────────────────── */}
      <div className="topbar">
        <div className="topbar-brand">
          <span className="brand-anchor">⚓</span>
          <h1>WRECKHUNTER <span className="brand-year">2000</span></h1>
        </div>
        <span
          className={`conn-badge ${online === true ? "online" : "offline"}`}
          onClick={handleRetryConnection}
          style={{ cursor: "pointer" }}
          title="Click to retry connection"
        >
          {online === null ? "…" : online ? "API Online" : "Mock Data"}
        </span>
        <span className="topbar-api-url" title="Resolved backend API base URL">
          {apiBase}
        </span>
        <input
          className="search-box"
          type="text"
          placeholder="Search wrecks by name…"
          value={searchQuery}
          onChange={e => {
            setSearchQuery(e.target.value);
            if (panel !== "list" && e.target.value.length >= 2) {
              setQuickSteelOnly(false);
              setQuickMagFilter("");
              setPanel("list");
            }
          }}
        />
      </div>

      {/* ── Sidebar ────────────────────────────────────── */}
      <div className="sidebar">
        <div className="section-label">Intelligence</div>
        <button className={panel === "search" ? "active" : ""} onClick={() => setPanel("search")}>
          🔍 Search & Rescue
        </button>
        <button className={panel === "stats" ? "active" : ""} onClick={() => setPanel("stats")}>
          📊 Dashboard
        </button>
        <button className={panel === "list" ? "active" : ""} onClick={() => setPanel("list")}>
          📋 Wreck Registry
        </button>
        <button className={panel === "map" ? "active" : ""} onClick={() => setPanel("map")}>
          🗺 Map
        </button>
        <button className={panel === "earth" ? "active" : ""} onClick={() => handleSetPanel("earth")}>
          🌍 Earth Monitor
        </button>

        {selectedWreck && (
          <button className={panel === "detail" ? "active" : ""} onClick={() => setPanel("detail")}>
            🔍 {selectedWreck.name || "Detail"}
          </button>
        )}

        <div className="section-label">Quick Filters</div>
        <button onClick={() => goListWithFilter(false, "")}>
          All Wrecks
        </button>
        <button onClick={() => goListWithFilter(true, "")}>
          🛳 Steel Freighters
        </button>
        <button onClick={() => goListWithFilter(false, "strong")}>
          🧲 Strong Magnetic
        </button>
        <button onClick={() => goListWithFilter(false, "moderate")}>
          〰 Moderate Magnetic
        </button>

        <div className="section-label">
          {unlocked ? "Agent 🔓" : "Agent 🔒"}
        </div>
        <button className={panel === "agent" ? "active" : ""} onClick={() => handleSetPanel("agent")}>
          🤖 AI Director
        </button>
        <button className={panel === "kobold" ? "active" : ""} onClick={() => handleSetPanel("kobold")}>
          🤖 Kobold Agent
        </button>
        <button className={panel === "workers" ? "active" : ""} onClick={() => handleSetPanel("workers")}>
          ⚙️ Node Status
        </button>
        <button className={panel === "idle" ? "active" : ""} onClick={() => handleSetPanel("idle")}>
          💤 Idle Controller
        </button>
        <button className={panel === "scan-output" ? "active" : ""} onClick={() => setPanel("scan-output")}>
          📍 Scan Output
        </button>

        <div className="section-label">
          {unlocked ? "Pipelines 🔓" : "Pipelines 🔒"}
        </div>
        <button className={panel === "scan" ? "active" : ""} onClick={() => handleSetPanel("scan")}>
          📡 Scanner & Restoration
        </button>
        <button className={panel === "mag" ? "active" : ""} onClick={() => handleSetPanel("mag")}>
          🧲 Mag Pipeline
        </button>
        <button className={panel === "erie" ? "active" : ""} onClick={() => handleSetPanel("erie")}>
          🌊 Lake Erie Scanner
        </button>
        <button className={panel === "mission" ? "active" : ""} onClick={() => handleSetPanel("mission")}>
          🎯 Mission Control
        </button>
        <button className={panel === "pdf" ? "active" : ""} onClick={() => handleSetPanel("pdf")}>
          📄 PDF Redactor
        </button>

        <div className="section-label">Analysis</div>
        <button className={panel === "satellite" ? "active" : ""} onClick={() => setPanel("satellite")}>
          🛰 Satellite Tracker
        </button>
        <button className={panel === "loran" ? "active" : ""} onClick={() => handleSetPanel("loran")}>
          🌍 Loran-C Warp
        </button>
        <button className={panel === "sensors" ? "active" : ""} onClick={() => handleSetPanel("sensors")}>
          🛰 Extended Sensors
        </button>
        <button className={panel === "harvest" ? "active" : ""} onClick={() => handleSetPanel("harvest")}>
          🪝 Raw Harvester
        </button>

        <div className="section-label">
          {unlocked ? "Export 🔓" : "Export 🔒"}
        </div>
        <button className={panel === "export" ? "active" : ""} onClick={() => handleSetPanel("export")}>
          📤 KML / KMZ Export
        </button>

        {!unlocked && (
          <button
            onClick={() => setShowGate(true)}
            style={{ marginTop: "0.8rem", opacity: 0.7, fontSize: "0.8rem" }}
          >
            🔐 Unlock Full Access
          </button>
        )}
        {unlocked && (
          <button
            onClick={() => { setUnlocked(false); sessionStorage.removeItem("wh2k_unlocked"); }}
            style={{ marginTop: "0.8rem", opacity: 0.6, fontSize: "0.8rem" }}
          >
            🔒 Lock
          </button>
        )}
      </div>

      {/* ── Main content ───────────────────────────────── */}
      <div className="main-content">
        <div style={{ display: panel === "search" ? "block" : "none", height: '100%' }}>
          <SearchPanel />
        </div>
        <div style={{ display: panel === "stats" ? "block" : "none", height: '100%' }}>
          <StatsPanel />
        </div>
        <div style={{ display: panel === "list" ? "block" : "none", height: '100%' }}>
          <WreckList
            key={`list-${listKey}`}
            searchQuery={searchQuery}
            onSelect={handleSelect}
            initialSteelOnly={quickSteelOnly}
            initialMagFilter={quickMagFilter}
          />
        </div>
        <div style={{ display: panel === "map" ? "block" : "none", height: '100%' }}>
          <MapPanel onSelect={handleSelect} />
        </div>
        <div style={{ display: panel === "earth" ? "block" : "none", height: '100%' }}>
          <EarthMonitorPanel />
        </div>

        <div style={{ display: panel === "workers" ? "block" : "none", height: '100%' }}>
          <WorkerStatusPanel />
        </div>
        <div style={{ display: panel === "scan" ? "block" : "none", height: '100%' }}>
          <ScanPanel />
        </div>
        <div style={{ display: panel === "mag" ? "block" : "none", height: '100%' }}>
          <MagPipelinePanel />
        </div>
        <div style={{ display: panel === "erie" ? "block" : "none", height: '100%' }}>
          <EriePanel />
        </div>
        <div style={{ display: panel === "restore" ? "block" : "none", height: '100%' }}>
          <RestorationPanel />
        </div>
        <div style={{ display: panel === "pdf" ? "block" : "none", height: '100%' }}>
          <PDFBreakerPanel />
        </div>
        <div style={{ display: panel === "export" ? "block" : "none", height: '100%' }}>
          <ExportPanel />
        </div>
        <div style={{ display: panel === "loran" ? "block" : "none", height: '100%' }}>
          <LoranPanel />
        </div>
        <div style={{ display: panel === "sensors" ? "block" : "none", height: '100%' }}>
          <ExtendedSensorsPanel />
        </div>
        <div style={{ display: panel === "harvest" ? "block" : "none", height: '100%' }}>
          <HarvesterPanel />
        </div>
        <div style={{ display: panel === "satellite" ? "block" : "none", height: '100%' }}>
          <SatelliteTrackerPanel />
        </div>
        <div style={{ display: panel === "agent" ? "block" : "none", height: '100%' }}>
          <AgentPanel />
        </div>
        <div style={{ display: panel === "kobold" ? "block" : "none", height: '100%' }}>
          <KoboldAgentPanel />
        </div>
        <div style={{ display: panel === "mission" ? "block" : "none", height: '100%' }}>
          <MissionControlPanel />
        </div>
        <div style={{ display: panel === "idle" ? "block" : "none", height: '100%' }}>
          <IdleControlPanel />
        </div>
        <div style={{ display: panel === "scan-output" ? "block" : "none", height: '100%' }}>
          <ScanOutputPanel />
        </div>
        <div style={{ display: panel === "detail" ? "block" : "none", height: '100%' }}>
          {selectedWreck && <WreckDetail wreck={selectedWreck} onBack={() => setPanel("list")} />}
        </div>
      </div>
    </div>
  );
}


