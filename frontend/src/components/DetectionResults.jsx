import React, { useState } from 'react'
import api, { BACKEND_URL } from '../services/api'

function annotatedImageUrl(path) {
  if (!path) return ''
  return /^https?:\/\//i.test(path) ? path : `${BACKEND_URL}${path.startsWith('/') ? '' : '/'}${path}`
}

export default function DetectionResults({ result }) {
  const [reportLoading, setReportLoading] = useState(false)
  const [reportMsg, setReportMsg] = useState('')

  if (!result) {
    return (
      <div className="card">
        <h3 style={{ marginTop: 0 }}>Detected Objects</h3>
        <p style={{ color: 'var(--muted)' }}>No analysis results yet. Upload and analyze media first.</p>
      </div>
    )
  }

  if (result.frame_results) {
    const event = result.event_assessment
    return (
      <div className="card">
        <h3 style={{ marginTop: 0 }}>Video Analysis Results</h3>
        <p style={{ color: 'var(--muted)', fontSize: 14 }}>
          Frames analyzed: {result.video_info?.frames_analyzed} |
          Total detections: {result.total_detections} |
          Unique objects: {(result.unique_objects || []).join(', ') || 'None'}
        </p>
        {event && (
          <div style={{
            margin: '12px 0', padding: 12, borderRadius: 8,
            background: event.requires_human_review ? '#3f1d1d' : '#0f172a',
            border: `1px solid ${event.requires_human_review ? 'var(--warning)' : 'var(--border)'}`,
            fontSize: 13
          }}>
            <b>{event.status.replaceAll('_', ' ')}</b>
            <div style={{ color: 'var(--muted)', marginTop: 4 }}>{event.summary}</div>
            {event.review_frames?.length > 0 && (
              <div style={{ marginTop: 6 }}>
                Review timestamps: {event.review_frames.map(f => `${f.timestamp_sec}s`).join(', ')}
              </div>
            )}
          </div>
        )}
        <table className="table">
          <thead>
            <tr>
              <th>Frame</th>
              <th>Time (s)</th>
              <th>Detections</th>
              <th>Objects</th>
            </tr>
          </thead>
          <tbody>
            {result.frame_results.map((fr, i) => (
              <tr key={i}>
                <td>{fr.frame_number}</td>
                <td>{fr.timestamp_sec}</td>
                <td>{fr.detection_count}</td>
                <td>{fr.detections.map(d => d.class).join(', ') || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  const detections = result.detections || []

  const generateQuickReport = async () => {
    setReportLoading(true)
    setReportMsg('')
    try {
      const res = await api.post('/api/report/generate', {
        case: {
          case_id: 'CASE-' + Date.now().toString().slice(-8),
          description: 'Quick forensic report from detection results',
        },
        detections: detections,
        risk: result.risk || {},
        annotated_image: result.annotated_image || null,
        evidence_id: result.evidence_id || result.evidence?.id || null,
      })
      setReportMsg('Report generated: ' + res.data.report_name)
      if (res.data.download_url) {
        const link = document.createElement('a')
        link.href = BACKEND_URL + res.data.download_url
        link.download = res.data.report_name
        link.target = '_blank'
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
      }
    } catch (e) {
      setReportMsg('Error: ' + (e.response?.data?.error || e.message))
    } finally {
      setReportLoading(false)
    }
  }

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
        <h3 style={{ margin: 0 }}>Detected Objects ({detections.length})</h3>
        <button
          className="primary"
          onClick={generateQuickReport}
          disabled={reportLoading || detections.length === 0}
          style={{ fontSize: 13 }}
        >
          {reportLoading ? 'Generating…' : 'Generate Forensic Report'}
        </button>
      </div>

      {reportMsg && (
        <p style={{
          marginTop: 10, fontSize: 13,
          color: reportMsg.startsWith('Error') ? 'var(--danger)' : 'var(--success)'
        }}>
          {reportMsg}
        </p>
      )}

      {detections.length === 0 ? (
        <p style={{ color: 'var(--muted)', marginTop: 12 }}>No objects detected above confidence threshold.</p>
      ) : (
        <table className="table" style={{ marginTop: 12 }}>
          <thead>
            <tr>
              <th>Class</th>
              <th>Category</th>
              <th>Confidence</th>
              <th>Bounding Box</th>
            </tr>
          </thead>
          <tbody>
            {detections.map((d, i) => (
              <tr key={i}>
                <td><b>{d.class}</b></td>
                <td>
                  <span className={'badge ' + (
                    d.category === 'Weapons' ? 'badge-high' :
                    d.category === 'Persons' ? 'badge-medium' : 'badge-low'
                  )}>
                    {d.category}
                  </span>
                </td>
                <td>{(d.confidence * 100).toFixed(1)}%</td>
                <td style={{ fontSize: 12, color: 'var(--muted)' }}>
                  [{d.bbox?.join(', ')}]
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {result.annotated_image && (
        <div style={{ marginTop: 16 }}>
          <h4>Annotated result</h4>
          <img src={annotatedImageUrl(result.annotated_image)} alt="Annotated detection result"
            style={{ maxWidth: '100%', maxHeight: 560, borderRadius: 8 }} />
        </div>
      )}

      {result.cnn_message && !result.cnn_available && (
        <p style={{ color: 'var(--warning)', fontSize: 13 }}>CNN unavailable; showing YOLO-only results. {result.cnn_message}</p>
      )}
      {result.cnn_preprocessing?.source && (
        <p style={{ color: 'var(--muted)', fontSize: 12 }}>CNN preprocessing: {result.cnn_preprocessing.source}</p>
      )}
      {result.comparison_available && (
        <details style={{ marginTop: 12 }}>
          <summary>Compare YOLO-only detections</summary>
          <p style={{ color: 'var(--muted)', fontSize: 13 }}>
            {(result.yolo_detections || []).map(d => `${d.class_name} ${(d.confidence * 100).toFixed(1)}% [${d.bbox.join(', ')}]`).join(' · ') || 'No detections'}
          </p>
        </details>
      )}

      {result.risk && (
        <div style={{ marginTop: 12, fontSize: 13, color: 'var(--muted)' }}>
          Risk: <b style={{ color: 'var(--text)' }}>{result.risk.overall_risk}/100</b>
          {result.risk.risk_label && ' — ' + result.risk.risk_label}
        </div>
      )}
    </div>
  )
}
