import React, { useState, useEffect } from 'react'
import api, { BACKEND_URL } from '../services/api'

/**
 * Case Reports – generate & download professional PDF forensic reports
 */
export default function CaseReports({ analysisResult }) {
  const [loading, setLoading] = useState(false)
  const [lastReport, setLastReport] = useState(null)
  const [error, setError] = useState('')
  const [caseId, setCaseId] = useState('CASE-' + Date.now().toString().slice(-8))
  const [description, setDescription] = useState(
    'Automated forensic analysis of crime scene evidence by AI Crime Scene Evidence Analyzer'
  )
  const [recentReports, setRecentReports] = useState([])

  const loadRecent = () => {
    api.get('/api/report/list')
      .then(res => setRecentReports(res.data.reports || []))
      .catch(() => {})
  }

  useEffect(() => {
    loadRecent()
  }, [lastReport])

  const downloadUrl = (path) => {
    // Prefer proxy if same origin, otherwise full backend URL
    if (path.startsWith('http')) return path
    return `${BACKEND_URL}${path}`
  }

  const downloadReport = async (path, filename) => {
    const response = await api.get(path, { responseType: 'blob' })
    const pdf = new Blob([response.data], { type: 'application/pdf' })
    const objectUrl = URL.createObjectURL(pdf)
    const link = document.createElement('a')
    link.href = objectUrl
    link.download = filename || 'forensic_report.pdf'
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000)
  }

  const generateReport = async () => {
    if (!analysisResult || !(analysisResult.detections || []).length) {
      // Still allow report with empty detections (risk-only)
      if (!analysisResult) {
        setError('Please analyze an image or video first (Media Upload tab).')
        return
      }
    }

    setLoading(true)
    setError('')

    try {
      const payload = {
        case: {
          case_id: caseId,
          description: description,
        },
        detections: analysisResult.detections || [],
        risk: analysisResult.risk || {},
        annotated_image: analysisResult.annotated_image || null,
        evidence_id: analysisResult.evidence_id || analysisResult.evidence?.id || null,
        file_name: analysisResult.file_name || null,
        analyzed_at: analysisResult.analyzed_at || null,
        video_result: analysisResult.frame_results ? analysisResult : null,
      }

      const res = await api.post('/api/report/generate', payload)
      setLastReport(res.data)

      if (res.data.download_url) {
        await downloadReport(res.data.download_url, res.data.report_name)
      }
    } catch (e) {
      console.error(e)
      setError(e.response?.data?.error || e.userMessage || e.message || 'Report generation failed')
    } finally {
      setLoading(false)
    }
  }

  const hasAnalysis = !!analysisResult

  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Forensic Case Reports</h3>
      <p style={{ color: 'var(--muted)', fontSize: 14, marginBottom: 16 }}>
        Generate a professional PDF report containing risk analysis, detected objects,
        annotated evidence image, and chain-of-custody log.
      </p>

      {/* Case metadata */}
      <div style={{ display: 'grid', gap: 12, marginBottom: 16, maxWidth: 520 }}>
        <div>
          <label style={{ fontSize: 12, color: 'var(--muted)' }}>Case ID</label>
          <input
            value={caseId}
            onChange={e => setCaseId(e.target.value)}
            style={{ width: '100%' }}
          />
        </div>
        <div>
          <label style={{ fontSize: 12, color: 'var(--muted)' }}>Case Description</label>
          <textarea
            value={description}
            onChange={e => setDescription(e.target.value)}
            rows={2}
            style={{ width: '100%', resize: 'vertical' }}
          />
        </div>
      </div>

      {!hasAnalysis ? (
        <p style={{ color: 'var(--muted)', fontSize: 14 }}>
          Analyze media first using the <b>Media Upload</b> tab, then return here to generate the report.
        </p>
      ) : (
        <div style={{ marginBottom: 16, fontSize: 13, color: 'var(--muted)' }}>
          <div>Detections ready: <b style={{ color: 'var(--text)' }}>{(analysisResult.detections || []).length}</b></div>
          <div>Risk score: <b style={{ color: 'var(--text)' }}>{analysisResult.risk?.overall_risk ?? '—'}</b>
            {analysisResult.risk?.risk_label && ` (${analysisResult.risk.risk_label})`}
          </div>
        </div>
      )}

      <button
        className="primary"
        onClick={generateReport}
        disabled={loading || !hasAnalysis}
        style={{ minWidth: 220 }}
      >
        {loading ? 'Generating PDF…' : 'Generate PDF Forensic Report'}
      </button>

      {error && (
        <p style={{ color: 'var(--danger)', fontSize: 13, marginTop: 12 }}>{error}</p>
      )}

      {lastReport && (
        <div style={{
          marginTop: 16, padding: 14, background: '#0f172a',
          borderRadius: 8, fontSize: 13, border: '1px solid var(--border)'
        }}>
          <div style={{ color: 'var(--success)', marginBottom: 6, fontWeight: 600 }}>
            ✓ Report generated successfully
          </div>
          <div><b>File:</b> {lastReport.report_name}</div>
          <a
            href={downloadUrl(lastReport.download_url)}
            onClick={event => {
              event.preventDefault()
              downloadReport(lastReport.download_url, lastReport.report_name)
                .catch(downloadError => setError(downloadError.message || 'Report download failed.'))
            }}
            style={{ color: 'var(--primary)', display: 'inline-block', marginTop: 8 }}
          >
            Download PDF again
          </a>
        </div>
      )}

      {/* Recent reports list */}
      {recentReports.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <h4 style={{ margin: '0 0 10px', fontSize: 14 }}>Recent Reports</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {recentReports.slice(0, 8).map(r => (
              <div key={r.name} style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '8px 12px', background: '#0f172a', borderRadius: 6, fontSize: 13
              }}>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{r.name}</span>
                <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexShrink: 0 }}>
                  <span style={{ color: 'var(--muted)', fontSize: 12 }}>{r.size_kb} KB</span>
                  <a href={downloadUrl(r.download_url)} onClick={event => {
                    event.preventDefault()
                    downloadReport(r.download_url, r.name)
                      .catch(downloadError => setError(downloadError.message || 'Report download failed.'))
                  }}
                     style={{ color: 'var(--primary)' }}>
                    Download
                  </a>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
