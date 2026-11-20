
import { useEffect, useState } from "react";

const API_URL = `${import.meta.env.VITE_API_URL || ""}/api/detection/assault/detections`;

export default function AssaultDetections() {
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadVideos() {
      try {
        const response = await fetch(API_URL);

        if (!response.ok) {
          throw new Error(`API request failed (${response.status})`);
        }

        const data = await response.json();
        const availableVideos = data.videos || [];

        setVideos(availableVideos);

        if (availableVideos.length > 0) {
          setSelectedVideo(availableVideos[0].video_name);
        }
      } catch (err) {
        setError(
          `${err.message}. Make sure the Flask backend is running.`
        );
      } finally {
        setLoading(false);
      }
    }

    loadVideos();
  }, []);

  useEffect(() => {
    if (!selectedVideo) {
      setResult(null);
      return;
    }

    let cancelled = false;

    async function loadSelectedVideo() {
      setLoading(true);
      setError("");
      setResult(null);

      try {
        const response = await fetch(
          `${API_URL}?video=${encodeURIComponent(selectedVideo)}`
        );

        if (!response.ok) {
          throw new Error(`Could not load video (${response.status})`);
        }

        const data = await response.json();

        if (!cancelled) {
          setResult(data);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadSelectedVideo();

    return () => {
      cancelled = true;
    };
  }, [selectedVideo]);

  const detections = result?.detections || [];

  function formatTime(seconds) {
    const safeSeconds = Number(seconds) || 0;
    const minutes = Math.floor(safeSeconds / 60);
    const remainingSeconds = (safeSeconds % 60).toFixed(2);

    return `${String(minutes).padStart(2, "0")}:${String(
      remainingSeconds
    ).padStart(5, "0")}`;
  }

  return (
    <section style={styles.container}>
      <h2 style={styles.heading}>Assault Video Detection</h2>
      <p style={styles.description}>
        Browse existing model predictions and their timestamps.
        These detections have not yet been verified against ground truth.
      </p>

      {error && <p style={styles.error}>{error}</p>}

      {loading && <p>Loading detection data...</p>}

      {!loading && !error && videos.length === 0 && (
        <p>No video results were returned by the backend.</p>
      )}

      {videos.length > 0 && (
        <>
          <label style={styles.label} htmlFor="assault-video">
            Select assault video
          </label>

          <select
            id="assault-video"
            value={selectedVideo}
            onChange={(event) => setSelectedVideo(event.target.value)}
            style={styles.select}
          >
            {videos.map((video) => (
              <option key={video.video_name} value={video.video_name}>
                {video.video_name} ({video.detection_count} detections)
              </option>
            ))}
          </select>

          {result && !loading && (
            <>
              <div style={styles.summary}>
                <div>
                  <strong>Video</strong>
                  <p>{result.video_name}</p>
                </div>

                <div>
                  <strong>Predicted detections</strong>
                  <p>{detections.length}</p>
                </div>
              </div>

              {detections.length === 0 ? (
                <p>No detections are listed for this video.</p>
              ) : (
                <div style={styles.tableWrapper}>
                  <table style={styles.table}>
                    <thead>
                      <tr>
                        <th style={styles.cell}>Timestamp</th>
                        <th style={styles.cell}>Frame</th>
                        <th style={styles.cell}>Class</th>
                        <th style={styles.cell}>Confidence</th>
                      </tr>
                    </thead>

                    <tbody>
                      {detections.map((detection, index) => (
                        <tr
                          key={`${detection.frame}-${detection.class_id}-${index}`}
                        >
                          <td style={styles.cell}>
                            {formatTime(detection.timestamp_seconds)}
                          </td>
                          <td style={styles.cell}>{detection.frame}</td>
                          <td style={styles.cell}>{detection.class}</td>
                          <td style={styles.cell}>
                            {detection.confidence == null
                              ? "—"
                              : `${(
                                  Number(detection.confidence) * 100
                                ).toFixed(1)}%`}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </>
      )}
    </section>
  );
}

const styles = {
  container: {
    padding: "24px",
    color: "var(--text-color, #222)",
    fontFamily: "Arial, sans-serif",
  },
  heading: {
    marginBottom: "8px",
  },
  description: {
    marginBottom: "20px",
    lineHeight: 1.5,
  },
  label: {
    display: "block",
    marginBottom: "8px",
    fontWeight: 600,
  },
  select: {
    width: "100%",
    maxWidth: "500px",
    padding: "10px",
    marginBottom: "20px",
  },
  summary: {
    display: "flex",
    flexWrap: "wrap",
    gap: "32px",
    margin: "16px 0",
  },
  tableWrapper: {
    width: "100%",
    overflowX: "auto",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    textAlign: "left",
  },
  cell: {
    padding: "10px",
    borderBottom: "1px solid #ddd",
  },
  error: {
    color: "#c62828",
    padding: "12px",
    background: "#ffebee",
    borderRadius: "6px",
  },
};
