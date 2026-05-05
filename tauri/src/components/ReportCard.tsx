// ReportCard — displays results from a completed search job.

/* eslint-disable @typescript-eslint/no-explicit-any */

export interface SearchReport {
  job_id: string;
  status: string;
  query: string;
  elapsed_seconds: number | null;
  spec: {
    target?: string;
    target_type?: string;
    area?: string;
    bbox?: number[];
    date_range?: string[];
    sensors?: string[];
    passes?: number[];
    sensitivity?: number;
  };
  detections: { count: number; top: any[] };
  wreck_db_matches: { count: number; top: any[] };
  output: Record<string, any>;
  idle_status: Record<string, any>;
}

export default function ReportCard({ report }: { report: SearchReport }) {
  const elapsed = report.elapsed_seconds;

  return (
    <div style={{ border: "1px solid #30363d", borderRadius: 10, background: "#161b22", padding: "1rem" }}>
      <div style={{ fontWeight: 600, color: "#e6edf3", marginBottom: "0.8rem", fontSize: 14 }}>
        📋 Report
      </div>

      {/* Summary counts */}
      <div style={{ display: "flex", gap: "1.5rem", marginBottom: "0.8rem" }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 28, fontWeight: 700, color: "#58a6ff" }}>{report.detections.count}</div>
          <div style={{ fontSize: 11, color: "#8b949e" }}>Detections</div>
        </div>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 28, fontWeight: 700, color: "#3fb950" }}>{report.wreck_db_matches.count}</div>
          <div style={{ fontSize: 11, color: "#8b949e" }}>DB Matches</div>
        </div>
        {elapsed !== null && (
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: "#d29922" }}>
              {elapsed < 60 ? `${elapsed}s` : `${Math.floor(elapsed / 60)}m`}
            </div>
            <div style={{ fontSize: 11, color: "#8b949e" }}>Elapsed</div>
          </div>
        )}
      </div>

      {/* Wreck DB matches */}
      {report.wreck_db_matches.top.length > 0 && (
        <div style={{ marginBottom: "0.8rem" }}>
          <div style={{ fontSize: 12, color: "#8b949e", marginBottom: 4 }}>Wreck DB Matches</div>
          {report.wreck_db_matches.top.map((m: any, i: number) => (
            <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid #21262d", fontSize: 12 }}>
              <span style={{ color: "#e6edf3" }}>{m.name ?? ""}</span>
              <span style={{ color: "#8b949e" }}>
                {m.distance_m != null ? `${Math.round(m.distance_m)}m` : ""}{" "}
                <span style={{ color: "#3fb950" }}>
                  {m.match_score != null ? `${(m.match_score * 100).toFixed(0)}%` : ""}
                </span>
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Top detections */}
      {report.detections.top.length > 0 && (
        <div>
          <div style={{ fontSize: 12, color: "#8b949e", marginBottom: 4 }}>Top Detections</div>
          {report.detections.top.slice(0, 5).map((d: any, i: number) => (
            <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid #21262d", fontSize: 12 }}>
              <span style={{ color: "#8b949e" }}>
                {d.latitude?.toFixed(4)}, {d.longitude?.toFixed(4)}
              </span>
              <span style={{ color: "#d29922" }}>{d.method ?? d.source_file ?? ""}</span>
              <span style={{ color: "#58a6ff" }}>
                {d.confidence != null ? `${(d.confidence * 100).toFixed(0)}%` : ""}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Pipeline stdout */}
      {report.output?.stdout && (
        <details style={{ marginTop: "0.6rem" }}>
          <summary style={{ fontSize: 12, color: "#8b949e", cursor: "pointer" }}>
            Pipeline output
          </summary>
          <pre style={{
            fontSize: 11, color: "#8b949e", background: "#0d1117",
            padding: "0.5rem", borderRadius: 4, overflow: "auto",
            maxHeight: 200, marginTop: 4,
          }}>
            {String(report.output.stdout).slice(-2000)}
          </pre>
        </details>
      )}
    </div>
  );
}
