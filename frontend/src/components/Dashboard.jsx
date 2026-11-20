import React, { useState, useEffect, useRef } from 'react'
import api, { BACKEND_URL } from '../services/api'
import MediaUpload from './MediaUpload'
import DetectionResults from './DetectionResults'
import RiskScore from './RiskScore'
import CaseReports from './CaseReports'
import Chatbot from './Chatbot'
import HistoryLogs from './HistoryLogs'
import SceneChange from './SceneChange'
import VideoAnalysis from './VideoAnalysis'
import EvidenceLibrary from './EvidenceLibrary'

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('overview')
  const [analysisResult, setAnalysisResult] = useState(null)
  const [summary, setSummary] = useState(null)
  const [summaryLoading, setSummaryLoading] = useState(true)
  const [summaryError, setSummaryError] = useState('')
  const [summaryVersion, setSummaryVersion] = useState(0)
  const [reportInventory, setReportInventory] = useState(null)
  const [reportLoading, setReportLoading] = useState(true)
  const [reportError, setReportError] = useState('')
  const [modelStatus, setModelStatus] = useState(null)
  const mediaPreviewUrl = useRef(null)
  const databaseWasUnavailable = useRef(false)

  useEffect(() => () => {
    if (mediaPreviewUrl.current) URL.revokeObjectURL(mediaPreviewUrl.current)
  }, [])

  useEffect(() => {
    setSummaryLoading(true)
    setSummaryError('')
    api.get('/api/dashboard/summary')
      .then(res => setSummary(res.data))
      .catch(err => setSummaryError(err.userMessage || err.response?.data?.message || err.response?.data?.error || 'Dashboard statistics could not be loaded.'))
      .finally(() => setSummaryLoading(false))
    setReportLoading(true)
    setReportError('')
    api.get('/api/report/list')
      .then(res => setReportInventory(res.data))
      .catch(err => setReportError(err.userMessage || err.response?.data?.error || 'Saved reports could not be loaded.'))
      .finally(() => setReportLoading(false))
    api.get('/api/detection/status')
      .then(res => setModelStatus(res.data))
      .catch(() => setModelStatus({ status: 'offline' }))
  }, [analysisResult, summaryVersion])

  // Continue checking while MongoDB is offline so the dashboard reloads the
  // persisted counts as soon as the backend reconnects.
  useEffect(() => {
    let active = true
    const checkDatabase = () => api.get('/api/dashboard/health')
      .then(response => {
        if (!active) return
        if (databaseWasUnavailable.current && response.data?.status === 'connected') {
          databaseWasUnavailable.current = false
          setSummaryVersion(version => version + 1)
        }
      })
      .catch(error => {
        if (active && (error.response?.status === 503 || error.code === 'ERR_NETWORK')) {
          databaseWasUnavailable.current = true
        }
      })
    checkDatabase()
    const timer = window.setInterval(checkDatabase, 15000)
    return () => { active = false; window.clearInterval(timer) }
  }, [])

  useEffect(() => {
    const handleNavigation = event => setActiveTab(event.detail)
    window.addEventListener('forensiq:navigate', handleNavigation)
    return () => window.removeEventListener('forensiq:navigate', handleNavigation)
  }, [])

  useEffect(() => {
    window.dispatchEvent(new CustomEvent('forensiq:tab-changed', { detail: activeTab }))
  }, [activeTab])

  const chatbotContext = {
    dashboard_summary: summary ? {
      status: summary.status,
      total_evidence: summary.total_evidence,
      total_cases: summary.total_cases,
      recent_evidence: (summary.recent_evidence || []).map(item => ({
        filename: item.filename,
        file_type: item.file_type,
        status: item.status,
        uploaded_at: item.uploaded_at,
        risk_score: item.risk_score?.overall_risk ?? item.risk_score,
        detections: (item.detections || []).slice(0, 5).map(detection => ({
          class: detection.class,
          category: detection.category,
          confidence: detection.confidence
        }))
      })),
      recent_logs: (summary.recent_logs || []).slice(0, 5).map(log => ({
        action: log.action || log.event,
        timestamp: log.timestamp || log.created_at
      }))
    } : null,
    current_analysis: analysisResult ? {
      detections: (analysisResult.detections || []).slice(0, 20).map(detection => ({
        class: detection.class,
        category: detection.category,
        confidence: detection.confidence
      })),
      risk: analysisResult.risk || null
    } : null
  }

  return (
    <div className="dashboard">
      <div className="dashboard-head">
        <div>
          <div className="eyebrow">Deep learning workspace / YOLOv8</div>
          <h1>{activeTab === 'evidence' ? 'Evidence Library' : activeTab === 'video' || (activeTab === 'results' && analysisResult?.frame_results) ? 'Video Analysis' : activeTab === 'upload' || activeTab === 'results' ? 'Image Analysis' : 'Forensic Dashboard'}</h1>
          <p className="dashboard-intro">{activeTab === 'evidence' ? 'Review saved category predictions and media.' : activeTab === 'video' || (activeTab === 'results' && analysisResult?.frame_results) ? 'Motion evidence Â· frame extraction pipeline' : activeTab === 'upload' || activeTab === 'results' ? 'Static evidence Â· object detection pipeline' : 'Investigation overview Â· evidence, risk, and chain of custody'}</p>
        </div>
        <div className="workspace-actions"><span className="secure-pill">Secure session</span><span className="profile-chip">{(summary?.user?.username || 'AI').slice(0, 2).toUpperCase()}</span></div>
      </div>

      {summary && activeTab !== 'overview' && activeTab !== 'dashboard' && (
        <div className="grid grid-3 metric-grid">
          <div className="card metric-card">
            <div className="metric-value">{summary.total_evidence ?? 'Unavailable'}</div>
            <div className="metric-label">Evidence records</div>
          </div>
          <div className="card metric-card">
            <div className="metric-value">{summary.total_cases ?? 'Unavailable'}</div>
            <div className="metric-label">Open case files</div>
          </div>
          <div className="card metric-card">
            <div className="metric-value">{summary.recent_logs?.length || 0}</div>
            <div className="metric-label">Recent custody events</div>
          </div>
        </div>
      )}

      {(activeTab === 'overview' || activeTab === 'dashboard') && (
        <>
          <OverviewDashboard summary={summary} loading={summaryLoading} error={summaryError} reportInventory={reportInventory} reportLoading={reportLoading} reportError={reportError} onRetry={() => setSummaryVersion(value => value + 1)} onNavigate={setActiveTab} />
          <CaseRegistration onCreated={() => setSummaryVersion(value => value + 1)} />
        </>
      )}

      {activeTab === 'evidence' && <EvidenceLibrary />}

      {(activeTab === 'upload' || activeTab === 'video') && (
        <>
          <MediaUpload onComplete={(res) => {
            if (mediaPreviewUrl.current) URL.revokeObjectURL(mediaPreviewUrl.current)
            mediaPreviewUrl.current = res.media_preview || null
            setAnalysisResult(res)
            setActiveTab('results')
          }} />
        </>
      )}

      {activeTab === 'results' && (
        <AnalysisWorkspace result={analysisResult} modelStatus={modelStatus} onUpload={() => setActiveTab('upload')} onReport={() => setActiveTab('reports')} />
      )}

      {activeTab === 'scene' && <SceneChange />}

      {activeTab === 'risk' && (
        <RiskScore risk={analysisResult?.risk} />
      )}

      {activeTab === 'reports' && (
        <CaseReports analysisResult={analysisResult} />
      )}

      {activeTab === 'history' && <HistoryLogs />}

      {activeTab === 'chatbot' && (
        <Chatbot context={chatbotContext} />
      )}
    </div>
  )
}

function CaseRegistration({ onCreated }) {
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [cases, setCases] = useState([])
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    api.get('/api/dashboard/cases')
      .then(response => setCases(response.data.cases || []))
      .catch(() => setError('Saved cases are unavailable while MongoDB is disconnected.'))
  }, [])

  const createCase = async event => {
    event.preventDefault()
    setBusy(true)
    setError('')
    setMessage('')
    try {
      const response = await api.post('/api/dashboard/cases', { title, description })
      const saved = response.data.case
      setCases(previous => [saved, ...previous])
      setTitle('')
      setDescription('')
      setMessage(`Saved case ID: ${saved.id}`)
      onCreated?.()
    } catch (requestError) {
      setError(requestError.userMessage || requestError.response?.data?.error || 'Case could not be saved.')
    } finally {
      setBusy(false)
    }
  }

  return <section className="card dashboard-record-section">
    <div className="panel-heading"><div><h2>Investigation cases</h2><p className="muted-copy">Create and view case records saved in MongoDB.</p></div></div>
    <form onSubmit={createCase} className="case-registration-form">
      <label>Case title<input required value={title} onChange={event => setTitle(event.target.value)} maxLength={160} /></label>
      <label>Description<textarea value={description} onChange={event => setDescription(event.target.value)} rows={2} maxLength={2000} /></label>
      <button className="primary" type="submit" disabled={busy}>{busy ? 'Saving…' : 'Register case'}</button>
    </form>
    {error && <p className="dashboard-state dashboard-error" role="alert">{error}</p>}
    {message && <p className="dashboard-state" role="status">{message}</p>}
    {cases.length > 0 && <div className="table-scroll"><table className="table evidence-table"><thead><tr><th>Case ID</th><th>Title</th><th>Status</th><th>Evidence</th><th>Risk level</th><th>Created</th></tr></thead><tbody>{cases.map(item => <tr key={item.id}><td>{item.id}</td><td>{item.title}</td><td>{item.status || 'Unknown'}</td><td>{item.evidence_count ?? 'Unavailable'}</td><td>{item.risk_level || 'Unknown'}</td><td>{item.created_at ? new Date(item.created_at).toLocaleString() : 'Unavailable'}</td></tr>)}</tbody></table></div>}
  </section>
}

function OverviewDashboard({ summary, loading, error, reportInventory, onRetry, onNavigate }) {
  const unavailable = summary?.status === 'unavailable'
  const value = count => count == null ? 'Unavailable' : Number(count).toLocaleString()
  const bands = [
    { name: 'Low', count: summary?.risk_distribution?.low },
    { name: 'Medium', count: summary?.risk_distribution?.moderate },
    { name: 'High', count: summary?.risk_distribution?.high },
    { name: 'Critical', count: summary?.risk_distribution?.critical },
  ]
  const metrics = [
    ['Total Cases', summary?.total_cases, 'Registered investigations', '▱'],
    ['High Risk Cases', summary?.high_risk_cases, 'Cases currently rated high or critical', '△'],
    ['Images Processed', summary?.images_analyzed, 'Successful image analyses', '▧'],
    ['Videos Processed', summary?.videos_analyzed, 'Successfully analyzed videos', '▷'],
    ['Reports Generated', summary?.reports_generated ?? reportInventory?.total_reports, 'Saved PDF reports', '▤'],
  ]
  const objectDistribution = summary?.object_distribution
  const expectedObjects = ['Person', 'Weapon', 'Gun', 'Hammer', 'Helmet', 'Knife', 'Mask', 'Pistol', 'Rifle']
  const objectRows = objectDistribution == null ? [] : [
    ...expectedObjects.map(name => [name, Object.entries(objectDistribution).reduce((total, [key, count]) => total + (key.toLowerCase() === name.toLowerCase() ? Number(count) || 0 : 0), 0)]),
    ['Other', Object.entries(objectDistribution).reduce((total, [key, count]) => total + (expectedObjects.some(name => name.toLowerCase() === key.toLowerCase()) ? 0 : Number(count) || 0), 0)],
  ]
  const objectTotal = objectRows.reduce((sum, [, count]) => sum + Number(count || 0), 0)
  let donutOffset = 0
  const donutStops = objectRows.map(([, count], index) => {
    const start = donutOffset
    donutOffset += objectTotal ? Number(count) / objectTotal * 100 : 0
    return `${['#1674ff', '#12d9e8', '#ff527d', '#f1b627', '#7b62ff', '#ff8b37', '#8ca6c8', '#42b8ff', '#23d69b', '#8391ad'][index % 10]} ${start}% ${donutOffset}%`
  }).join(', ')
  const recentCases = summary?.recent_evidence || []
  const recentCustody = summary?.recent_custody || summary?.recent_logs || []
  return <div className="forensic-overview">
    {loading && <div className="dashboard-state" role="status">Loading investigation statistics…</div>}
    {error && <div className="dashboard-state dashboard-error" role="alert">{error} <button className="text-button" onClick={onRetry}>Retry</button></div>}
    {!loading && !error && summary?.status === 'partial' && summary.message && <div className="dashboard-state" role="status">{summary.message}</div>}
    {!loading && unavailable && <div className="dashboard-state">{summary.message || 'Dashboard data is unavailable.'}</div>}

    <section className="dashboard-metrics screenshot-metrics">
      {metrics.map(([label, count, caption, icon], index) => <article className={`overview-metric card metric-tone-${index}`} key={label}><span className="metric-icon" aria-hidden="true">{icon}</span><span>{label}</span><strong className={loading ? 'dashboard-skeleton' : ''}>{loading ? '\u00a0' : value(count)}</strong><small>{caption}</small></article>)}
    </section>

    {!error && <>
      <section className="overview-panels">
        <article className="card overview-chart-card"><div className="panel-heading"><div><h2>Object Detection Distribution</h2><p className="muted-copy">YOLO detections from successful saved analyses</p></div><button className="text-button" onClick={() => onNavigate('results')}>View detections →</button></div>
          {objectTotal > 0 ? <div className="object-distribution"><div className="object-donut" style={{ background: `conic-gradient(${donutStops})` }} role="img" aria-label={`${value(objectTotal)} detections`}><div><strong>{value(objectTotal)}</strong><small>detections</small></div></div><div className="chart-legend">{objectRows.map(([name, count], index) => <div className="legend-row" key={`${name}-${index}`} title={`${name}: ${value(count)}`}><i className={`legend-color color-${index}`} /><span>{name}</span><b>{value(count)}</b></div>)}</div></div> : <div className="overview-empty">{loading ? 'Loading saved detections…' : unavailable || error ? 'Detection data is unavailable.' : 'No saved object detections yet.'}</div>}
        </article>
        <article className="card overview-chart-card"><div className="panel-heading"><div><h2>Risk analysis</h2><p className="muted-copy">Risk bands across saved evidence</p></div><button className="text-button" onClick={() => onNavigate('risk')}>Review risk ?</button></div>
          {loading ? <div className="overview-empty">Loading risk data…</div> : summary?.risk_assessments > 0 && bands.every(band => band.count != null) ? <RiskAnalysisChart bands={bands} maximum={Math.max(1, ...bands.map(band => Number(band.count) || 0))} /> : <div className="overview-empty">{unavailable || summary?.risk_assessments == null ? 'Risk data is unavailable.' : 'No saved risk assessments yet.'}</div>}
        </article>
      </section>

      <section className="card monthly-cases-card"><div className="panel-heading"><div><h2>Monthly Cases</h2><p className="muted-copy">Cases created over the last seven months</p></div></div><MonthlyEvidenceChart data={summary?.monthly_evidence} loading={loading} /></section>

      <section className="card recent-evidence-card"><div className="panel-heading"><div><h2>Recent Cases</h2><p className="muted-copy">Latest persisted evidence records</p></div><button className="text-button" onClick={() => onNavigate('history')}>View case history →</button></div>
        {recentCases.length ? <div className="table-scroll"><table className="table evidence-table"><thead><tr><th>Case ID</th><th>Title</th><th>Type</th><th>Date</th><th>Risk</th><th>Threat</th><th>Status</th><th>Action</th></tr></thead><tbody>{recentCases.map(item => {
          const risk = item.risk_score && typeof item.risk_score === 'object' ? item.risk_score : null
          const score = risk?.overall_risk ?? risk?.score ?? item.risk_score
          const numericScore = Number(score)
          const thresholds = summary?.risk_thresholds || { moderate: 25, high: 50, critical: 75 }
          const threat = risk?.risk_level || risk?.threat_level || (score == null || !Number.isFinite(numericScore) ? 'Unavailable' : numericScore >= thresholds.critical ? 'Critical' : numericScore >= thresholds.high ? 'High' : numericScore >= thresholds.moderate ? 'Medium' : 'Low')
          return <tr key={item.id}><td>{item.case_id || 'Unassigned'}</td><td title={item.filename}>{item.filename || 'Unnamed evidence'}</td><td>{item.file_type || 'Unavailable'}</td><td>{item.uploaded_at ? new Date(item.uploaded_at).toLocaleDateString() : 'Unavailable'}</td><td>{score == null ? 'Unavailable' : score}</td><td><span className={`risk-tag ${String(threat).toLowerCase()}`}>{threat}</span></td><td><span className={`status-tag ${String(item.status || '').toLowerCase()}`}>{item.status || 'Unavailable'}</span></td><td><button className="text-button" onClick={() => onNavigate('evidence')}>View</button></td></tr>
        })}</tbody></table></div> : <div className="overview-empty">{loading ? 'Loading recent cases…' : unavailable || error ? 'Recent cases are unavailable.' : 'No evidence records have been saved.'}</div>}
      </section>

      <section className="card recent-evidence-card"><div className="panel-heading"><div><h2>Recent custody activity</h2><p className="muted-copy">Persisted chain-of-custody events</p></div><button className="text-button" onClick={() => onNavigate('history')}>Open logs →</button></div>
        {loading ? <div className="overview-empty">Loading custody activity…</div> : summary?.custody_status !== 'available' ? <div className="overview-empty">Custody records are unavailable because the database connection is {summary?.custody_status === 'error' ? 'in an error state' : 'offline'}.</div> : recentCustody.length ? <div className="table-scroll"><table className="table evidence-table"><thead><tr><th>Evidence ID</th><th>Case</th><th>Action</th><th>User</th><th>Timestamp</th><th>Status</th></tr></thead><tbody>{recentCustody.slice(0, 8).map(item => <tr key={item.id}><td>{item.evidence_id || 'Unavailable'}</td><td>{item.case || item.case_id || item.filename || 'Unavailable'}</td><td>{item.action || item.event || 'Custody event'}</td><td>{item.user_id || 'Unavailable'}</td><td>{item.timestamp ? new Date(item.timestamp).toLocaleString() : 'Unavailable'}</td><td>{item.record_status || 'Recorded'}</td></tr>)}</tbody></table></div> : <div className="overview-empty">No custody events have been recorded.</div>}
      </section>

    </>}
  </div>
}

function MonthlyEvidenceChart({ data, loading }) {
  if (loading) return <div className="overview-empty">Loading monthly cases…</div>
  if (data == null) return <div className="overview-empty">Monthly case data is unavailable.</div>
  const values = Array.isArray(data) ? data : []
  const maximum = Math.max(5, ...values.map(item => Number(item.case_count) || 0))
  const points = values.map((item, index) => ({
    ...item,
    x: values.length <= 1 ? 340 : 52 + index * (580 / (values.length - 1)),
    casesY: 166 - ((Number(item.case_count) || 0) / maximum) * 124,
  }))
  const casesLine = points.map(point => `${point.x},${point.casesY}`).join(' ')

  return <div className="monthly-chart-wrap">
    {points.length ? <svg className="monthly-chart" viewBox="0 0 680 220" role="img" aria-label="Cases created per month">
      {[0, 1, 2, 3, 4, 5].map(step => {
        const y = 166 - step * 24
        return <g key={step}><line x1="52" x2="648" y1={y} y2={y} className="chart-grid-line" /><text x="36" y={y + 4} className="chart-axis-label" textAnchor="end">{Math.round(maximum * step / 5)}</text></g>
      })}
      <polyline points={casesLine} className="monthly-case-line" />
      {points.map(point => <g key={point.month}><circle cx={point.x} cy={point.casesY} r="4" className="monthly-case-point"><title>{point.month}: {point.case_count} cases created</title></circle><text x={point.x} y="197" className="chart-axis-label" textAnchor="middle">{point.month}</text></g>)}
    </svg> : <div className="overview-empty">Monthly activity trend will appear when records are available.</div>}
    {points.length > 0 && <div className="monthly-legend"><span><i className="monthly-case-key" /> Cases created</span></div>}
  </div>
}

function ClassificationDistribution({ data }) {
  const names = ['Assault', 'Robbery', 'Shooting', 'Normal']
  const colors = ['#945bff', '#12cde8', '#ff4f80', '#19d69a']
  const rows = names.map(name => [name, Number(data[name] || 0)])
  const total = rows.reduce((sum, [, count]) => sum + count, 0)
  let offset = 0
  const gradient = rows.map(([, count], index) => {
    const start = offset
    offset += total ? count / total * 100 : 0
    return `${colors[index]} ${start}% ${offset}%`
  }).join(', ')
  return <div className="classification-distribution">
    <div className="classification-donut" style={{ background: `conic-gradient(${gradient})` }} role="img" aria-label={`${total} classified videos`}>
      <div><strong>{total.toLocaleString()}</strong><small>videos</small></div>
    </div>
    <div className="chart-legend classification-legend">{rows.map(([name, count], index) => <div className="legend-row" key={name} title={`${name}: ${count}`}><i style={{ background: colors[index] }} /><span>{name}</span><b>{count.toLocaleString()}</b></div>)}</div>
  </div>
}

function RiskAnalysisChart({ bands, maximum }) {
  return <svg className="risk-analysis-chart" viewBox="0 0 680 250" role="img" aria-label="Cases by risk band">
    {[0, 1, 2, 3, 4, 5].map(step => {
      const y = 198 - step * 31
      return <g key={step}><line x1="48" x2="660" y1={y} y2={y} className="chart-grid-line" /><text x="34" y={y + 4} className="chart-axis-label" textAnchor="end">{Math.round(maximum * step / 5)}</text></g>
    })}
    {bands.map((band, index) => {
      const x = 94 + index * 145
      const height = (band.count / maximum) * 155
      return <g key={band.name}><rect x={x} y={198 - height} width="88" height={height} rx="8" className="risk-chart-bar"><title>{band.name}: {band.count} risk assessments</title></rect><text x={x + 44} y="224" className="chart-axis-label" textAnchor="middle">{band.name}</text><text x={x + 44} y={Math.max(39, 190 - height)} className="chart-value-label" textAnchor="middle">{band.count}</text></g>
    })}
  </svg>
}

function formatTimestamp(seconds) {
  const value = Math.max(0, Math.floor(Number(seconds) || 0))
  return `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(value % 60).padStart(2, '0')}`
}

function AnalysisWorkspace({ result, modelStatus, onUpload, onReport }) {
  if (result?.frame_results) return <VideoAnalysis result={result} onUpload={onUpload} onReport={onReport} />

  if (!result) {
    return (
      <section className="analysis-empty card">
        <div className="section-kicker">YOLOv8 image analysis</div>
        <h2>Upload evidence to begin.</h2>
        <p className="muted-copy">The trained deep-learning model will identify objects, calculate confidence, annotate the image, and produce a forensic risk score.</p>
        <button className="primary" onClick={onUpload}>Choose evidence -&gt;</button>
      </section>
    )
  }

  if (result.frame_results) {
    const classification = result.classification || result.event_classification
    const timeline = result.timeline || []
    return (
      <div className="analysis-workspace">
        <section className="card">
          <div className="section-kicker">Whole-video event classification Â· trained video model</div>
          <h2>{classification?.label || 'Classification unavailable'}</h2>
          <p>Confidence: {classification ? `${(classification.confidence * 100).toFixed(1)}%` : 'â€”'}</p>
          {classification?.probabilities && <p className="muted-copy">{Object.entries(classification.probabilities).map(([name, value]) => `${name}: ${(value * 100).toFixed(1)}%`).join(' Â· ')}</p>}
          <p className="muted-copy">{classification?.classifier_note}</p>
        </section>
        <section className="card">
          <div className="section-kicker">Sampled-frame visual evidence Â· YOLO</div>
          <p>Frames analyzed: {result.video_info?.frames_analyzed ?? 0} Â· Detections: {result.total_detections ?? 0} Â· Objects: {(result.unique_objects || []).join(', ') || 'None detected'}</p>
          {!result.video_info?.timestamps_available && <p className="muted-copy">The video FPS is unavailable or invalid, so reliable evidence timestamps could not be calculated.</p>}
          {timeline.length > 0 && <table className="table"><thead><tr><th>Timestamp</th><th>Frame</th><th>Evidence</th><th>YOLO confidence</th></tr></thead><tbody>{timeline.map((item, index) => <tr key={`${item.frame_number}-${index}`}><td>{item.timestamp_sec == null ? 'Unavailable' : `${item.timestamp_sec.toFixed(1)} s`}</td><td>{item.frame_number}</td><td>{item.class_name || item.class}</td><td>{(item.confidence * 100).toFixed(1)}%</td></tr>)}</tbody></table>}
        </section>
        <section className="card"><h3>Investigation summary</h3><p>{result.investigation_summary}</p><small>{result.disclaimer}</small></section>
      </div>
    )
  }

  const detections = result.detections || []
  const score = result.risk?.overall_risk ?? 0
  const thresholds = result.risk?.thresholds || { moderate: 20, elevated: 45, critical: 75 }
  const imageName = result.annotated_image?.split(/[/\\]/).pop()
  const annotatedUrl = imageName ? `${BACKEND_URL}/api/detection/annotated/${imageName}` : null
  const recommendations = detections.filter(item => item.category === 'Weapons').length
    ? ['Secure the perimeter and preserve the scene.', 'Collect the detected object for evidence handling.', 'Review all annotated detections before reporting.']
    : ['Review the annotated objects and confidence values.', 'Preserve the original media for case documentation.', 'Generate a report after investigator review.']

  return (
    <div className="analysis-workspace">
      <section className="analysis-score card">
        <div><div className="section-kicker">AI risk assessment</div><div className="score-line"><strong>{score}</strong><span>/ 100</span></div></div>
        <div className="score-meter"><span style={{ width: `${Math.min(score, 100)}%` }} /></div>
        <span className={`risk-pill ${score >= thresholds.critical ? 'high' : score >= thresholds.moderate ? 'medium' : 'low'}`}>Threat level: {result.risk?.threat_level || result.risk?.risk_label || 'Low'}</span>
        {result.risk?.reasons?.length > 0 && <ul className="muted-copy">{result.risk.reasons.map((reason, index) => <li key={index}>{reason}</li>)}</ul>}
        <small className="muted-copy">{result.risk?.score_disclaimer}</small>
      </section>

      <div className="evidence-grid">
        <section className="card evidence-card">
          <div className="panel-heading"><h2>AI annotated</h2><span className="count-pill">{detections.length} objects</span></div>
          {annotatedUrl ? <img className="evidence-image" src={annotatedUrl} alt="YOLOv8 annotated evidence" /> : <div className="image-placeholder">Annotated output unavailable</div>}
          <small className="model-caption">Model: {modelStatus?.model_path || result.model_used || 'YOLOv8'}</small>
        </section>
        <section className="card detections-card">
          <div className="panel-heading"><h2>Detections</h2><span className="panel-note">Confidence</span></div>
          {detections.length ? <div className="detection-list">{detections.slice(0, 8).map((item, index) => <div className="detection-row" key={`${item.class}-${index}`}><div><b>{item.class}</b><small>{item.category}</small></div><div className="confidence"><span><i style={{ width: `${item.confidence * 100}%` }} /></span><b>{Math.round(item.confidence * 100)}%</b></div></div>)}</div> : <p className="muted-copy">No objects detected above the configured confidence threshold.</p>}
        </section>
      </div>

      <section className="card recommendations-card">
        <div className="panel-heading"><h2>Recommendations</h2><span className="panel-note">Generated from YOLOv8 evidence</span></div>
        <div className="recommendation-list">{recommendations.map((item, index) => <div className="recommendation-row" key={item}><span>{index + 1}</span><p>{item}</p></div>)}</div>
      </section>
    </div>
  )
}
