import { useEffect, useRef, useState } from "react";
import "./App.css";

type Cartridge = {
  id: string;
  name: string;
  value: number;
};

type WindowInfo = {
  id: number;
  title: string;
};

type StateResponse = {
  status: string;
  is_running: boolean;
  cartridges: Record<string, number>;
  percentages: Record<string, number>;
  pwm: Record<string, number>;
  latest: {
    screenshot_size: { width: number; height: number };
    should_send: boolean;
    clip_sources?: Record<string, number>;
    sources?: Record<string, number>;
    object_areas?: Record<string, number>;
    scene_areas?: Record<string, number>;
    metrics?: any;
  } | null;
  clip_sources?: Record<string, number>;
  sources?: Record<string, number>;
  object_areas?: Record<string, number>;
  scene_areas?: Record<string, number>;
  metrics?: any;
};

type LogEntry = {
  id: number;
  source: string;
  message: string;
};

type PerformanceMetrics = {
  capture_time_ms: number;
  process_time_ms?: number;
  segmentation_time_ms: number;
  classification_time_ms: number;
  total_time_ms: number;
  fps: number;
  timestamp: string;
};

type ExtendedStateResponse = StateResponse & {
  clip_sources?: Record<string, number>;
  sources?: Record<string, number>;
  object_areas?: Record<string, number>;
  scene_areas?: Record<string, number>;
  metrics?: {
    capture_time_ms: number;
    process_time_ms?: number;
    segmentation_time_ms: number;
    classification_time_ms: number;
    detection_time_ms: number;
    total_time_ms: number;
    fps: number;
    timestamp: string;
  };
};

const getApiUrl = () => {
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return '';
  }
  const savedIp = localStorage.getItem('backend_ip');
  if (savedIp) {
    return `http://${savedIp}:8000`;
  }
  const currentHost = window.location.hostname;
  return `http://${currentHost}:8000`;
};

const API_URL = getApiUrl();
const WS_URL = API_URL ? API_URL.replace('http', 'ws') + '/ws' : '/ws';
const POLL_INTERVAL_MS = 5000;

const initialCartridges: Cartridge[] = [
  { id: "pine", name: "Pine", value: 0 },
  { id: "earth", name: "Earth", value: 0 },
  { id: "ocean", name: "Ocean", value: 0 },
  { id: "smoke", name: "Smoke", value: 0 },
  { id: "rain", name: "Rain", value: 0 },
  { id: "asphalt", name: "Asphalt", value: 0 },
];

function App() {
  const [isRunning, setIsRunning] = useState(false);
  const [isBackendReady, setIsBackendReady] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [backendIp, setBackendIp] = useState<string>(() => localStorage.getItem('backend_ip') || '');
  const [cartridges, setCartridges] = useState<Cartridge[]>(initialCartridges);
  const [fps, setFps] = useState(0);
  const [screenshotSize, setScreenshotSize] = useState({ width: 0, height: 0 });
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [clipSources, setClipSources] = useState<Record<string, number>>({});
  const [, setYoloObjects] = useState<Record<string, number>>({});

  const [windows, setWindows] = useState<WindowInfo[]>([]);
  const [selectedWindowId, setSelectedWindowId] = useState<number | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([
    { id: 1, source: "system", message: "Application ready" },
    { id: 2, source: "capture", message: "Waiting for screen capture module" },
    { id: 3, source: "ai", message: "Waiting for Python engine" },
  ]);

  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    capture_time_ms: 0,
    process_time_ms: 0,
    segmentation_time_ms: 0,
    classification_time_ms: 0,
    total_time_ms: 0,
    fps: 0,
    timestamp: '',
  });
  const [metricsHistory, setMetricsHistory] = useState<PerformanceMetrics[]>([]);
  const [avgMetrics, setAvgMetrics] = useState({
    avg_capture: 0,
    avg_process: 0,
    avg_segmentation: 0,
    avg_classification: 0,
    avg_total: 0,
  });

  const intervalRef = useRef<number | null>(null);
  const logIdRef = useRef(4);
  const lastPollTimeRef = useRef<number | null>(null);
  const previewUrlRef = useRef<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const isMountedRef = useRef(true);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const metricsHistoryRef = useRef<PerformanceMetrics[]>([]);

  function addLog(source: string, message: string) {
    const newLog: LogEntry = {
      id: logIdRef.current,
      source,
      message,
    };
    logIdRef.current += 1;
    setLogs((currentLogs) => {
      const updatedLogs = [...currentLogs, newLog];
      return updatedLogs.slice(-12);
    });
  }

  const cleanupWebSocket = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch (e) {
        // ignore
      }
      wsRef.current = null;
    }
  };

  const cleanupIntervals = () => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const setupWebSocket = () => {
    if (!isMountedRef.current || !isRunning) {
      return;
    }
    cleanupWebSocket();
    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => {
        if (!isMountedRef.current) return;
        addLog("system", "WebSocket connected");
      };
      ws.onmessage = (event) => {
        if (!isMountedRef.current) return;
        try {
          const data = JSON.parse(event.data);
          if (data.clip_sources) {
            setClipSources(data.clip_sources);
          } else if (data.sources) {
            setClipSources(data.sources);
          }
          if (data.object_areas) {
            setYoloObjects(data.object_areas);
          } else if (data.scene_areas) {
            setYoloObjects(data.scene_areas);
          }
          if (data.cartridges) {
            setCartridges((current) =>
              current.map((cartridge) => ({
                ...cartridge,
                value: Math.max(0, Math.min(1, data.cartridges[cartridge.id] ?? 0)),
              }))
            );
          }
          if (data.screenshot_size) {
            setScreenshotSize(data.screenshot_size);
          }
          if (data.metrics) {
            updateMetrics(data.metrics);
          }
          loadPreview();
        } catch (error) {
          console.error('WebSocket message error:', error);
        }
      };
      ws.onerror = (error) => {
        if (!isMountedRef.current) return;
        console.error('WebSocket error:', error);
        addLog("error", "WebSocket connection error");
      };
      ws.onclose = () => {
        if (!isMountedRef.current) return;
        addLog("system", "WebSocket disconnected");
        if (isMountedRef.current && isRunning) {
          reconnectTimeoutRef.current = setTimeout(() => {
            setupWebSocket();
          }, 3000);
        }
      };
    } catch (error) {
      console.error('WebSocket setup error:', error);
      if (isMountedRef.current && isRunning) {
        reconnectTimeoutRef.current = setTimeout(() => {
          setupWebSocket();
        }, 5000);
      }
    }
  };

  const updateMetrics = (newMetrics: PerformanceMetrics) => {
    setMetrics(newMetrics);
    const history = [...metricsHistoryRef.current, newMetrics];
    if (history.length > 50) {
      history.shift();
    }
    metricsHistoryRef.current = history;
    setMetricsHistory(history);
    if (history.length > 0) {
      const avg = history.reduce((acc, m) => ({
        avg_capture: acc.avg_capture + m.capture_time_ms,
        avg_process: acc.avg_process + (m.process_time_ms || (m.segmentation_time_ms + m.classification_time_ms)),
        avg_segmentation: acc.avg_segmentation + m.segmentation_time_ms,
        avg_classification: acc.avg_classification + m.classification_time_ms,
        avg_total: acc.avg_total + m.total_time_ms,
      }), { avg_capture: 0, avg_process: 0, avg_segmentation: 0, avg_classification: 0, avg_total: 0 });
      setAvgMetrics({
        avg_capture: avg.avg_capture / history.length,
        avg_process: avg.avg_process / history.length,
        avg_segmentation: avg.avg_segmentation / history.length,
        avg_classification: avg.avg_classification / history.length,
        avg_total: avg.avg_total / history.length,
      });
    }
  };

  async function checkBackend() {
    if (!isMountedRef.current) return;
    try {
      const response = await fetch(`${API_URL}/health`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      const ready = data.status === "ok" && data.models_loaded === true && data.capture_ready === true;
      if (!isMountedRef.current) return;
      setIsBackendReady(ready);
      if (ready) {
        addLog("ai", "Python engine connected");
        addLog("capture", "Screen capture connected");

        // *** ДОДАНО: завантажуємо список вікон ***
        void fetchWindows();

        try {
          const ipResponse = await fetch(`${API_URL}/ip`);
          const ipData = await ipResponse.json();
          if (ipData.ip && ipData.ip !== '127.0.0.1') {
            setBackendIp(ipData.ip);
            localStorage.setItem('backend_ip', ipData.ip);
          }
        } catch (e) {
          console.log('Could not get backend IP');
        }
      }
    } catch {
      if (!isMountedRef.current) return;
      setIsBackendReady(false);
      addLog("error", "Backend connection failed");
    }
  }

  async function loadPreview() {
    if (!isMountedRef.current) return;
    try {
      const response = await fetch(`${API_URL}/preview?t=${Date.now()}`, {
        cache: "no-store",
      });
      if (!response.ok) return;
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      if (previewUrlRef.current !== null) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
      previewUrlRef.current = objectUrl;
      setPreviewUrl(objectUrl);
    } catch (error) {
      console.error('Preview load error:', error);
    }
  }

  async function pollState() {
    if (!isMountedRef.current || !isRunning) return;
    try {
      setIsAnalyzing(true);
      const startTime = performance.now();
      const response = await fetch(`${API_URL}/state`);
      if (!response.ok) {
        const errorBody = await response.text();
        throw new Error(`State request failed: ${response.status} ${errorBody}`);
      }
      const data = await response.json() as ExtendedStateResponse;
      if (!isMountedRef.current) return;

      if (data.clip_sources) {
        setClipSources(data.clip_sources);
      } else if (data.latest?.clip_sources) {
        setClipSources(data.latest.clip_sources);
      } else if (data.sources) {
        setClipSources(data.sources);
      } else if (data.latest?.sources) {
        setClipSources(data.latest.sources);
      }

      if (data.object_areas) {
        setYoloObjects(data.object_areas);
      } else if (data.scene_areas) {
        setYoloObjects(data.scene_areas);
      }

      if (data.latest) {
        setScreenshotSize(data.latest.screenshot_size);
      }

      setCartridges((currentCartridges) =>
        currentCartridges.map((cartridge) => ({
          ...cartridge,
          value: Math.max(0, Math.min(1, data.cartridges[cartridge.id] ?? 0)),
        }))
      );

      await loadPreview();
      const endTime = performance.now();
      const totalTime = endTime - startTime;
      const now = Date.now();
      if (lastPollTimeRef.current !== null) {
        const elapsedSeconds = (now - lastPollTimeRef.current) / 1000;
        if (elapsedSeconds > 0) {
          const currentFps = 1 / elapsedSeconds;
          setFps(currentFps);
        }
      }
      lastPollTimeRef.current = now;
      const activeCartridges = Object.entries(data.percentages || {})
        .filter(([, value]) => value > 0)
        .map(([name, value]) => `${name}: ${value}%`);
      if (activeCartridges.length > 0) {
        addLog("ai", `Detected ${activeCartridges.join(", ")}`);
      }
      updateMetrics({
        capture_time_ms: totalTime * 0.3,
        process_time_ms: totalTime * 0.6,
        segmentation_time_ms: totalTime * 0.3,
        classification_time_ms: totalTime * 0.3,
        total_time_ms: totalTime,
        fps: 1000 / totalTime,
        timestamp: new Date().toLocaleTimeString(),
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown analysis error";
      addLog("error", message);
      setIsBackendReady(false);
    } finally {
      if (isMountedRef.current) {
        setIsAnalyzing(false);
      }
    }
  }

  async function startAnalysis() {
    if (!isMountedRef.current) return;
    if (!isBackendReady) {
      await checkBackend();
    }
    try {
      const response = await fetch(`${API_URL}/start`, {
        method: "POST",
      });
      if (!response.ok) {
        const errorBody = await response.text();
        throw new Error(`Start request failed: ${response.status} ${errorBody}`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not start analysis";
      addLog("error", message);
      return;
    }
    setIsRunning(true);
    addLog("runtime", "Analysis started");
    setTimeout(() => {
      if (isMountedRef.current && isRunning) {
        setupWebSocket();
      }
    }, 500);
    await pollState();
  }

  async function stopAnalysis() {
    setIsRunning(false);
    cleanupIntervals();
    cleanupWebSocket();
    try {
      await fetch(`${API_URL}/stop`, {
        method: "POST",
      });
    } catch {
      addLog("error", "Could not send stop command");
    }
    setCartridges(initialCartridges);
    setFps(0);
    lastPollTimeRef.current = null;
    addLog("runtime", "Analysis stopped");
  }

  // Отримання списку відкритих вікон
  async function fetchWindows() {
    if (!isMountedRef.current) return;
    try {
      const response = await fetch(`${API_URL}/windows`);
      if (!response.ok) throw new Error("Failed to fetch windows");
      const data = await response.json();
      setWindows(data.windows || data || []);
    } catch (error) {
      console.error("Error fetching windows:", error);
      addLog("error", "Failed to load window list");
    }
  }

  // Відправка вибраного window_id на бекенд
  async function selectWindow(windowId: number) {
    try {
      const response = await fetch(`${API_URL}/windows/select/${windowId}`, {
        method: "POST",
      });
      if (!response.ok) throw new Error("Failed to select window");
      setSelectedWindowId(windowId);
      addLog("system", `Selected window ID: ${windowId}`);
    } catch (error) {
      console.error("Error selecting window:", error);
      addLog("error", `Could not select window ${windowId}`);
    }
  }

  const handleIpChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const ip = e.target.value;
    setBackendIp(ip);
    localStorage.setItem('backend_ip', ip);
  };

  const reconnectWithNewIp = () => {
    window.location.reload();
  };

  useEffect(() => {
    isMountedRef.current = true;
    void checkBackend();
    return () => {
      isMountedRef.current = false;
      cleanupIntervals();
      cleanupWebSocket();
      if (previewUrlRef.current !== null) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!isRunning) {
      cleanupIntervals();
      return;
    }
    intervalRef.current = window.setInterval(() => {
      pollState();
    }, POLL_INTERVAL_MS);
    pollState();
    return () => {
      cleanupIntervals();
    };
  }, [isRunning]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">AI GAMING SCENT SYSTEM</p>
          <h1>Admin Panel</h1>
          {/* {backendIp && <span className="backend-ip">📡 {backendIp}</span>} */}
        </div>
        <div className={`status ${isRunning ? "running" : "stopped"}`}>
          <span className="status-dot" />
          {isRunning ? "Running" : "Stopped"}
        </div>
      </header>

      {/* {window.innerWidth < 768 && (
        <div className="ip-config">
          <input
            type="text"
            value={backendIp}
            onChange={handleIpChange}
            placeholder="Enter backend IP (e.g., 192.168.1.100)"
          />
          <button onClick={reconnectWithNewIp}>Connect</button>
        </div>
      )} */}

      <section className="dashboard-grid">
        <article className="panel preview-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-label">SCREEN CAPTURE</p>
              <h2>Game preview</h2>
            </div>
            <span className="fps">{fps.toFixed(1)} FPS</span>
          </div>
          <div className="preview-container">
            {previewUrl ? (
              <>
                <img className="preview-image" src={previewUrl} alt="Current screen capture" />
                {isAnalyzing && (
                  <div className="preview-overlay">
                    <span className="preview-spinner" />
                    Analyzing
                  </div>
                )}
                <div className="preview-resolution">
                  {screenshotSize.width} × {screenshotSize.height}
                </div>
              </>
            ) : (
              <div className="preview-placeholder">
                <div className="preview-icon">{isAnalyzing ? "◌" : "◉"}</div>
                {isBackendReady ? (
                  <>
                    <p>{isAnalyzing ? "Capturing current screen" : "Screen capture connected"}</p>
                    <span>Press Start analysis to capture a frame</span>
                  </>
                ) : (
                  <>
                    <p>Screen capture is not connected</p>
                    <span>Start FastAPI on port 8000</span>
                  </>
                )}
              </div>
            )}
          </div>
        </article>

        <article className="panel controls-panel">
          <p className="panel-label">RUNTIME</p>
          <h2>Engine control</h2>
          <div className="window-selector" style={{ marginBottom: "1rem" }}>
            <label style={{ display: "block", marginBottom: "0.5rem", fontSize: "0.85rem", opacity: 0.8 }}>
              Target Window:
            </label>
            <select
              value={selectedWindowId ?? ""}
              onChange={(e) => void selectWindow(Number(e.target.value))}
              disabled={isRunning || windows.length === 0}
              style={{
                width: "100%",
                padding: "0.5rem",
                borderRadius: "6px",
                background: "#1a1a1a",
                color: "#fff",
                border: "1px solid #333"
              }}
            >
              <option value="" disabled>
                {windows.length > 0 ? "Select a window to capture..." : "No windows available"}
              </option>
              {windows.map((win) => (
                <option key={win.id} value={win.id}>
                  {win.title} (ID: {win.id})
                </option>
              ))}
            </select>
          </div>
          <div className="control-buttons">
            <button
              className="primary-button"
              onClick={() => void startAnalysis()}
              disabled={isRunning || isAnalyzing || !isBackendReady}
            >
              {isAnalyzing ? "Analyzing..." : "Start analysis"}
            </button>
            <button
              className="secondary-button"
              onClick={() => void stopAnalysis()}
              disabled={!isRunning}
            >
              Stop
            </button>
          </div>
          <div className="runtime-info">
            <div>
              <span>Capture interval</span>
              <strong>2 seconds</strong>
            </div>
            <div>
              <span>AI engine</span>
              <strong>
                {isBackendReady ? "Connected" : <span className="status-error">Not connected</span>}
              </strong>
            </div>
          </div>
        </article>
      </section>

      {/* <section className="panel metrics-panel">
        <div className="panel-heading">
          <div>
            <p className="panel-label">PERFORMANCE</p>
            <h2>Processing Metrics</h2>
          </div>
          <div className="metrics-meta">
            <span className="metrics-fps">
              {metrics?.fps ? `${metrics.fps.toFixed(1)} FPS` : '0 FPS'}
            </span>
            <span className="metrics-timestamp">
              {metrics?.timestamp && ` | ${metrics.timestamp}`}
            </span>
          </div>
        </div>

        <div className="metrics-grid">
           
          <div className="metric-card">
            <span className="metric-label">Capture</span>
            <span className="metric-value">
              {metrics?.capture_time_ms ? metrics.capture_time_ms.toFixed(1) : '0.0'} ms
            </span>
            <span className="metric-avg">
              avg: {avgMetrics?.avg_capture ? avgMetrics.avg_capture.toFixed(1) : '0.0'} ms
            </span>
          </div>

           
          <div className="metric-card">
            <span className="metric-label">AI Process</span>
            <span className="metric-value">
              {metrics?.process_time_ms ? metrics.process_time_ms.toFixed(1) : '0.0'} ms
            </span>
            <span className="metric-avg">
              avg: {avgMetrics?.avg_process ? avgMetrics.avg_process.toFixed(1) : '0.0'} ms
            </span>
          </div>

         
          <div className="metric-card total">
            <span className="metric-label">Total Time</span>
            <span className="metric-value">
              {metrics?.total_time_ms ? metrics.total_time_ms.toFixed(1) : '0.0'} ms
            </span>
            <span className="metric-avg">
              avg: {avgMetrics?.avg_total ? avgMetrics.avg_total.toFixed(1) : '0.0'} ms
            </span>
          </div>
        </div>
 
        {metricsHistory && metricsHistory.length > 0 && (
          <div className="metrics-chart">
            <div className="chart-bars">
              {metricsHistory.slice(-20).map((m, i) => {
                const totalMs = m.total_time_ms || 0;
                const heightPercent = Math.min((totalMs / 500) * 100, 100);

                return (
                  <div
                    key={i}
                    className="chart-bar"
                    style={{
                      height: `${Math.max(heightPercent, 5)}%`,
                      background: totalMs > 250 ? '#ff6b6b' : totalMs > 100 ? '#fcc419' : '#51cf66'
                    }}
                    title={`${totalMs.toFixed(1)} ms (${m.timestamp || ''})`}
                  />
                );
              })}
            </div>
            <span className="chart-label">Last 20 frames performance history</span>
          </div>
        )}
      </section> */}

      <section className="panel cartridges-panel">
        <div className="panel-heading">
          <div>
            <p className="panel-label">OUTPUT</p>
            <h2>Cartridge intensity</h2>
          </div>
          <span className="cartridge-count">{cartridges.length} cartridges</span>
        </div>
        <div className="cartridge-grid">
          {cartridges.map((cartridge) => {
            const percentage = Math.round(cartridge.value * 100);
            return (
              <article className="cartridge-card" key={cartridge.id}>
                <div className="cartridge-header">
                  <span>{cartridge.name}</span>
                  <strong>{percentage}%</strong>
                </div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${percentage}%` }} />
                </div>
              </article>
            );
          })}
        </div>
      </section>

      <section className="panel clip-panel">
        <div className="panel-heading">
          <div>
            <p className="panel-label">CLIP ANALYSIS</p>
            <h2>Scene recognition</h2>
          </div>
          <span className="clip-count">
            {Object.keys(clipSources).filter(k => clipSources[k] > 0.05).length} active
          </span>
        </div>
        <div className="clip-grid">
          {Object.entries(clipSources)
            .filter(([, value]) => value > 0.05)
            .sort(([, a], [, b]) => b - a)
            .map(([name, score]) => {
              const percentage = Math.round(score * 100);
              return (
                <article className="clip-card" key={name}>
                  <div className="clip-header">
                    <span>{name.replace(/_/g, ' ').toUpperCase()}</span>
                    <strong>{percentage}%</strong>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill"
                      style={{
                        width: `${percentage}%`,
                        background: percentage > 50 ? '#51cf66' : percentage > 25 ? '#ffd43b' : '#ff6b6b'
                      }}
                    />
                  </div>
                  <span className="clip-score">score: {score.toFixed(3)}</span>
                </article>
              );
            })}
          {Object.keys(clipSources).filter(k => clipSources[k] > 0.05).length === 0 && (
            <div className="clip-placeholder">
              <p>No active sources detected</p>
              <span>Waiting for CLIP analysis...</span>
            </div>
          )}
        </div>
      </section>

      <section className="panel logs-panel">
        <div className="panel-heading">
          <div>
            <p className="panel-label">SYSTEM</p>
            <h2>Runtime log</h2>
          </div>
        </div>
        <div className="log-window">
          {logs.map((log) => (
            <p key={log.id}>
              <span>[{log.source}]</span> <strong>{log.message}</strong>
            </p>
          ))}
        </div>
      </section>
    </main>
  );
}

export default App;