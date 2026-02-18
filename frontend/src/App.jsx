import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import "./App.css";

const API = import.meta.env.VITE_BACKEND_HTTP;

export default function App() {
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [rawUrl, setRawUrl] = useState("");
  const [summary, setSummary] = useState(null);

  useEffect(() => {
    if (!file) {
      setRawUrl("");
      return;
    }
    const url = URL.createObjectURL(file);
    setRawUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const progress = status?.progress ?? 0;
  const phase = status?.message || (jobId ? "Starting..." : "");

  const outputs = status?.outputs || {};

  const safeOutputUrl = (key) => {
    const p = outputs?.[key];
    if (!p) return "";
    return `${API}${p}`;
  };

  const upload = async () => {
    setError("");
    setSummary(null);

    if (!file) {
      setError("Please select a video first.");
      return;
    }

    try {
      setIsUploading(true);
      setStatus(null);
      setJobId(null);

      const form = new FormData();
      form.append("file", file);

      const res = await axios.post(`${API}/api/upload`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setJobId(res.data.job_id);
    } catch (e) {
      setError("Upload failed. Check backend is running on port 8000.");
    } finally {
      setIsUploading(false);
    }
  };

  useEffect(() => {
    if (!jobId) return;

    const t = setInterval(async () => {
      try {
        const res = await axios.get(`${API}/api/status/${jobId}`);
        setStatus(res.data);

        if (res.data.status === "complete" || res.data.status === "error") {
          clearInterval(t);
        }
      } catch (e) {
        setError("Failed to fetch status.");
        clearInterval(t);
      }
    }, 900);

    return () => clearInterval(t);
  }, [jobId]);

  useEffect(() => {
    if (status?.status !== "complete" || !jobId) return;

    (async () => {
      try {
        const res = await fetch(`${API}/api/download/${jobId}/summary.json`);
        const data = await res.json();
        setSummary(data);
      } catch {
        // ignore
      }
    })();
  }, [status?.status, jobId]);

  const topStats = useMemo(() => {
    if (!summary) return null;
    const tracks  = summary?.counts?.unique_tracks ?? "-";
    const alerts  = summary?.counts?.alerts ?? 0;
    const windows = summary?.counts?.windows ?? "-";
    const grid    = summary?.video?.grid_size_px ?? "-";
    const stride  = summary?.video?.frame_stride ?? "-";
    return { tracks, alerts, windows, grid, stride };
  }, [summary]);

  const lastAlert = useMemo(() => {
    const arr = summary?.alerts || [];
    if (!arr.length) return null;
    return arr[arr.length - 1];
  }, [summary]);

  return (
    <div className="appShell">

      <header className="topBar">
        <div className="brand">
          <div className="brandMark">WS</div>
          <div>
            <div className="brandTitle">WareSpy</div>
            <div className="brandSub">Movement Analytics Platform</div>
          </div>
        </div>

        <div className="topBarMeta">
          <span className="liveChip">System Online</span>
        </div>
      </header>

      <main className="content">

        <section className="panel">
          <div className="panelHeader">
            <div className="sectionBadge">Input</div>
            <h2>
              <span className="panelIcon">🎬</span>
              Video Upload
            </h2>
          </div>

          <div className="divider" />

          <div className="uploader">
            <div className="filePick">
              <input
                type="file"
                accept="video/*"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
              <button
                className="primary"
                onClick={upload}
                disabled={!file || isUploading}
              >
                {isUploading ? "Uploading…" : "▶ Upload & Analyze"}
              </button>
            </div>

            {rawUrl && (
              <div className="rawPreview">
                <div className="rawTitle">Preview</div>
                <video src={rawUrl} controls />
                <div className="muted small" style={{ marginTop: 6 }}>
                  {file?.name} &nbsp;·&nbsp; {(file?.size / (1024 * 1024)).toFixed(1)} MB
                </div>
              </div>
            )}

            {jobId && (
              <div className="statusCard">
                <div className="statusTop">
                  <div>
                    <div className="label">Job ID</div>
                    <div className="mono">{jobId}</div>
                  </div>

                  <div className="statusPill">
                    <span
                      className={
                        status?.status === "complete"
                          ? "pill complete"
                          : status?.status === "error"
                          ? "pill error"
                          : "pill running"
                      }
                    >
                      {status?.status || "starting"}
                    </span>
                  </div>
                </div>

                <div className="barWrap">
                  <div className="barFill" style={{ width: `${progress}%` }} />
                </div>

                <div className="statusMsg">{phase}</div>

                {status?.error && (
                  <div className="errorBox">{status.error}</div>
                )}

                {topStats && (
                  <div className="statsRow">
                    <Stat label="Tracks"  value={topStats.tracks}  />
                    <Stat label="Alerts"  value={topStats.alerts}  />
                    <Stat label="Windows" value={topStats.windows} />
                    <Stat label="Grid px" value={topStats.grid}    />
                    <Stat label="Stride"  value={topStats.stride}  />
                  </div>
                )}

                {lastAlert && (
                  <div className="alertBox">
                    <div className="alertTitle">Latest Alert</div>
                    <div className="alertText">{lastAlert.message}</div>
                    <div className="muted small">
                      t={lastAlert.time_sec}s &nbsp;·&nbsp;
                      congested cells={lastAlert.congested_cells} &nbsp;·&nbsp;
                      max density={lastAlert.max_cell_density}
                    </div>
                  </div>
                )}
              </div>
            )}

            {error && <div className="errorBox">{error}</div>}
          </div>
        </section>

        <section className="panel">
          <div className="panelHeader">
            <div className="sectionBadge">Output</div>
            <h2>
              <span className="panelIcon">📊</span>
              Analysis Results
            </h2>
          </div>

          <div className="divider" />

          {status?.status !== "complete" ? (
            <div className="emptyState">
              <div className="emptyIcon">🔍</div>
              <div className="emptyTitle">No Results</div>
              <div className="muted">
                Upload and analyze a video to view outputs
              </div>
            </div>
          ) : (
            <>
              <div className="grid">
                <ResultCard
                  title="Real-time Heatmap Video"
                  type="video"
                  url={safeOutputUrl("heatmap_overlay.mp4")}
                />
                <ResultCard
                  title="Trajectory Overlay Video"
                  type="video"
                  url={safeOutputUrl("trajectory_overlay.mp4")}
                />
                <ResultCard
                  title="Cumulative Heatmap"
                  type="img"
                  url={safeOutputUrl("heatmap.png")}
                />
                <ResultCard
                  title="Speed Pace (Normalized)"
                  type="img"
                  url={safeOutputUrl("speed_chart.png")}
                />
                <ResultCard
                  title="Congestion Recurrence Overlay"
                  type="img"
                  url={safeOutputUrl("congestion_overlay.png")}
                />
              </div>

              <div className="downloadRow">
                <DownloadBtn
                  label="summary.json"
                  href={`${API}${outputs["summary.json"]}`}
                />
                <DownloadBtn
                  label="trajectory_overlay.mp4"
                  href={`${API}${outputs["trajectory_overlay.mp4"]}`}
                />
                <DownloadBtn
                  label="heatmap_overlay.mp4"
                  href={`${API}${outputs["heatmap_overlay.mp4"]}`}
                />
                <DownloadBtn
                  label="heatmap.png"
                  href={`${API}${outputs["heatmap.png"]}`}
                />
                <DownloadBtn
                  label="speed_chart.png"
                  href={`${API}${outputs["speed_chart.png"]}`}
                />
                <DownloadBtn
                  label="congestion_overlay.png"
                  href={`${API}${outputs["congestion_overlay.png"]}`}
                />
              </div>

              {summary?.bottlenecks_top5?.length ? (
                <div className="panelInner">
                  <div className="innerTitle">Bottleneck Detection</div>
                  <div className="bottleGrid">
                    {summary.bottlenecks_top5.map((b) => (
                      <div key={b.rank} className="bottleCard">
                        <div className="bottleRank">#{b.rank}</div>
                        <div className="bottleText">{b.note}</div>
                        <div className="muted small" style={{ marginTop: 5 }}>
                          Cell ({b.grid_cell.row},{b.grid_cell.col}) &nbsp;·&nbsp; {b.congested_windows} windows
                        </div>
                        <div className="muted small">
                          Position ({b.approx_location_px.x},{b.approx_location_px.y})
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="panelInner">
                  <div className="innerTitle">Bottleneck Detection</div>
                  <div className="muted small">
                    No congestion zones detected
                  </div>
                </div>
              )}
            </>
          )}
        </section>

      </main>

      <footer className="footer">
        <span className="muted small">
          WareSpy Analytics Engine
        </span>
        <span className="footerRight">v2.1.0</span>
      </footer>

    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="stat">
      <div className="statLabel">{label}</div>
      <div className="statValue">{value}</div>
    </div>
  );
}

function DownloadBtn({ label, href }) {
  if (!href) return null;
  return (
    <a className="dlBtn" href={href} target="_blank" rel="noreferrer">
      {label}
    </a>
  );
}

function ResultCard({ title, type, url }) {
  if (!url) {
    return (
      <div className="resultCard">
        <div className="resultTitle">{title}</div>
        <div className="muted small">Not available</div>
      </div>
    );
  }

  return (
    <div className="resultCard">
      <div className="resultTitle">{title}</div>
      {type === "img" ? (
        <img src={url} alt={title} />
      ) : (
        <video src={url} controls preload="metadata" />
      )}
    </div>
  );
}