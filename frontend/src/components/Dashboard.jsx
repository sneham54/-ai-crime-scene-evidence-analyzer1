import React, { useState, useEffect } from 'react'
import api, { BACKEND_URL } from '../services/api'
import MediaUpload from './MediaUpload'
import DetectionResults from './DetectionResults'
import RiskScore from './RiskScore'
import CaseReports from './CaseReports'
import Chatbot from './Chatbot'
import HistoryLogs from './HistoryLogs'
import SceneChange from './SceneChange'
import VideoAnalysis from './VideoAnalysis'

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('overview')
  const [analysisResult, setAnalysisResult] = useState(null)
  const [summary, setSummary] = useState(null)
  const [modelStatus, setModelStatus] = useState(null)

  useEffect(() => {
    api.get('/api/dashboard/summary')
      .then(res => setSummary(res.data))
      .catch(() => {})
    api.get('/api/detection/status')
      .then(res => setModelStatus(res.data))
      .catch(() => setModelStatus({ status: 'offline' }))
  }, [analysisResult])

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
          <h1>{activeTab === 'video' || (activeTab === 'results' && analysisResult?.frame_results) ? 'Video Analysis' : activeTab === 'upload' || activeTab === 'results' ? 'Image Analysis' : 'Forensic Dashboard'}</h1>
          <p className="dashboard-intro">{activeTab === 'video' || (activeTab === 'results' && analysisResult?.frame_results) ? 'Motion evidence · frame extraction pipeline' : activeTab === 'upload' || activeTab === 'results' ? 'Static evidence · object detection pipeline' : 'Investigation overview · evidence, risk, and chain of custody'}</p>
        </div>
        <div className="workspace-actions"><span className="secure-pill">Secure session</span><span className="profile-chip">{(summary?.user?.username || 'AI').slice(0, 2).toUpperCase()}</span></div>
      </div>

      {summary && activeTab !== 'overview' && activeTab !== 'dashboard' && (
        <div className="grid grid-3 metric-grid">
          <div className="card metric-card">
            <div className="metric-value">{summary.total_evidence || 0}</div>
            <div className="metric-label">Evidence records</div>
          </div>
          <div className="card metric-card">
            <div className="metric-value">{summary.total_cases || 0}</div>
            <div className="metric-label">Open case files</div>
          </div>
          <div className="card metric-card">
            <div className="metric-value">{summary.recent_logs?.length || 0}</div>
            <div className="metric-label">Recent custody events</div>
          </div>
        </div>
      )}

      {(activeTab === 'overview' || activeTab === 'dashboard') && (
        <OverviewDashboard summary={summary} onNavigate={setActiveTab} />
      )}

      {(activeTab === 'upload' || activeTab === 'video') && (
        <MediaUpload onComplete={(res) => {
          setAnalysisResult(res)
          setActiveTab('results')
        }} />
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

function OverviewDashboard({ summary, onNavigate }) {
  const evidence = summary?.recent_evidence || []
  const objectCounts = summary?.object_distribution || {}
  const allObjects = Object.entries(objectCounts).sort((a, b) => b[1] - a[1])
  const objects = allObjects.slice(0, 4)
  const otherCount = allObjects.slice(4).reduce((total, [, count]) => total + count, 0)
  if (otherCount) objects.push(['Other', otherCount])
  const detectionCount = allObjects.reduce((total, [, count]) => total + count, 0)
  const chartColors = ['#2864f0', '#31b8ec', '#f04452', '#f5a000', '#728096']
  let chartPosition = 0
  const donutStops = objects.map(([, count], index) => {
    const start = chartPosition
    chartPosition += (count / Math.max(detectionCount, 1)) * 100
    return `${chartColors[index]} ${start}% ${chartPosition}%`
  }).join(', ')
  const riskTotals = summary?.risk_distribution || {}
  const bands = [
    { name: 'Low', count: riskTotals.low || 0, color: 'low' },
    { name: 'Medium', count: riskTotals.medium || 0, color: 'medium' },
    { name: 'High', count: riskTotals.high || 0, color: 'high' },
    { name: 'Critical', count: riskTotals.critical || 0, color: 'critical' },
  ]
  const maxBand = Math.max(1, ...bands.map(item => item.count))
  const metrics = [
    ['Total cases', summary?.total_cases || 0, 'Registered case files'],
    ['High risk cases', summary?.high_risk_cases || 0, 'Risk score 70 or higher'],
    ['Images processed', summary?.images_analyzed || 0, 'Analyzed image evidence'],
    ['Videos processed', summary?.videos_analyzed || 0, 'Analyzed video evidence'],
    ['Reports generated', summary?.reports_generated || 0, 'PDF investigation reports'],
  ]

  return (
    <div className="forensic-overview">
      <section className="dashboard-metrics">
        {metrics.map(([label, value, hint]) => <article className="overview-metric card" key={label}><span>{label}</span><strong>{value}</strong><small>{hint}</small></article>)}
      </section>
      <section className="overview-panels">
        <article className="card overview-chart-card">
          <div className="panel-heading"><div><h2>Object detection distribution</h2><p className="muted-copy">From the recent evidence records</p></div><button className="text-button" onClick={() => onNavigate('results')}>View detections →</button></div>
          {objects.length ? <div className="object-distribution">
            <div className="object-donut" style={{ '--donut-background': `conic-gradient(${donutStops})` }}><div><strong>{summary?.total_detections || 0}</strong><small>detections</small></div></div>
            <div className="chart-legend">{objects.map(([name, count], index) => <div className="legend-row" key={name}><i className={`legend-color color-${index}`} /><span>{name}</span><b>{count}</b></div>)}</div>
          </div> : <div className="overview-empty">Object distribution will appear after evidence has been analyzed.</div>}
        </article>
        <article className="card overview-chart-card">
          <div className="panel-heading"><div><h2>Risk analysis</h2><p className="muted-copy">Risk bands across recent evidence</p></div><button className="text-button" onClick={() => onNavigate('risk')}>Review risk →</button></div>
          <RiskAnalysisChart bands={bands} maximum={maxBand} />
        </article>
      </section>
      <section className="card monthly-cases-card">
        <div className="panel-heading"><div><h2>Monthly cases</h2><p className="muted-copy">Evidence added over the last seven months</p></div></div>
        <MonthlyEvidenceChart data={summary?.monthly_evidence || []} />
      </section>
      <section className="card recent-evidence-card">
        <div className="panel-heading"><div><h2>Recent cases</h2><p className="muted-copy">Latest evidence linked to investigation records</p></div><button className="text-button" onClick={() => onNavigate('history')}>View case history</button></div>
        {evidence.length ? <div className="table-scroll"><table className="table evidence-table"><thead><tr><th>Case ID</th><th>Title</th><th>Type</th><th>Date</th><th>Risk</th><th>Threat</th><th>Status</th></tr></thead><tbody>{evidence.map((item, index) => {
          const risk = evidenceRisk(item)
          const level = risk >= 70 ? 'high' : risk >= 40 ? 'medium' : 'low'
          const caseId = item.case_id && item.case_id !== 'default-case' ? item.case_id : `EV-${String(item.id || index).slice(-5).toUpperCase()}`
          return <tr key={item.id || `${item.filename}-${index}`}><td><b>{caseId}</b></td><td>{item.filename || `Evidence ${index + 1}`}</td><td>{item.file_type || 'Media'}</td><td>{item.uploaded_at ? new Date(item.uploaded_at).toLocaleDateString() : '?'}</td><td>{risk}</td><td><span className={`risk-tag ${level}`}>{level}</span></td><td>{item.status || 'Recorded'}</td></tr>
        })}</tbody></table></div> : <div className="overview-empty">No evidence has been analyzed yet. Upload an image or video to begin.</div>}
      </section>
      <section className="card overview-activity-card">
        <div className="panel-heading"><div><h2>Recent custody activity</h2><p className="muted-copy">Chain-of-custody events</p></div><button className="text-button" onClick={() => onNavigate('history')}>Open logs →</button></div>
        {summary?.recent_logs?.length ? <div className="overview-activity-list">{summary.recent_logs.slice(0, 4).map((log, index) => <div className="overview-activity-row" key={index}><i /><span><b>{log.action || log.event || 'Evidence event'}</b><small>{log.timestamp || log.created_at || 'Recently recorded'}</small></span></div>)}</div> : <div className="overview-empty">Custody activity will show here when evidence is uploaded or analyzed.</div>}
      </section>
    </div>
  )
}

function evidenceRisk(item) {
  const risk = item?.risk_score?.overall_risk ?? item?.risk_score?.score ?? item?.risk_score ?? 0
  return Number.isFinite(Number(risk)) ? Number(risk) : 0
}

function MonthlyEvidenceChart({ data }) {
  const values = data.length ? data : []
  const maximum = Math.max(5, ...values.map(item => Number(item.count) || 0))
  const points = values.map((item, index) => ({
    ...item,
    x: values.length <= 1 ? 340 : 52 + index * (580 / (values.length - 1)),
    y: 166 - ((Number(item.count) || 0) / maximum) * 124,
  }))
  const line = points.map(point => `${point.x},${point.y}`).join(' ')
  const area = points.length ? `52,166 ${line} ${points[points.length - 1].x},166` : ''

  return <div className="monthly-chart-wrap">
    {points.length ? <svg className="monthly-chart" viewBox="0 0 680 220" role="img" aria-label="Monthly evidence counts">
      {[0, 1, 2, 3, 4, 5].map(step => {
        const y = 166 - step * 24
        return <g key={step}><line x1="52" x2="648" y1={y} y2={y} className="chart-grid-line" /><text x="36" y={y + 4} className="chart-axis-label" textAnchor="end">{Math.round(maximum * step / 5)}</text></g>
      })}
      <polygon points={area} className="monthly-area" />
      <polyline points={line} className="monthly-line" />
      {points.map(point => <g key={point.month}><circle cx={point.x} cy={point.y} r="4" className="monthly-point" /><text x={point.x} y="197" className="chart-axis-label" textAnchor="middle">{point.month}</text></g>)}
    </svg> : <div className="overview-empty">Monthly evidence trend will appear when uploads are recorded.</div>}
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
      return <g key={band.name}><rect x={x} y={198 - height} width="88" height={height} rx="8" className="risk-chart-bar" /><text x={x + 44} y="224" className="chart-axis-label" textAnchor="middle">{band.name}</text><text x={x + 44} y={Math.max(39, 190 - height)} className="chart-value-label" textAnchor="middle">{band.count}</text></g>
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
          <div className="section-kicker">Whole-video event classification · trained video model</div>
          <h2>{classification?.label || 'Classification unavailable'}</h2>
          <p>Confidence: {classification ? `${(classification.confidence * 100).toFixed(1)}%` : '—'}</p>
          {classification?.probabilities && <p className="muted-copy">{Object.entries(classification.probabilities).map(([name, value]) => `${name}: ${(value * 100).toFixed(1)}%`).join(' · ')}</p>}
          <p className="muted-copy">{classification?.classifier_note}</p>
        </section>
        <section className="card">
          <div className="section-kicker">Sampled-frame visual evidence · YOLO</div>
          <p>Frames analyzed: {result.video_info?.frames_analyzed ?? 0} · Detections: {result.total_detections ?? 0} · Objects: {(result.unique_objects || []).join(', ') || 'None detected'}</p>
          {!result.video_info?.timestamps_available && <p className="muted-copy">The video FPS is unavailable or invalid, so reliable evidence timestamps could not be calculated.</p>}
          {timeline.length > 0 && <table className="table"><thead><tr><th>Timestamp</th><th>Frame</th><th>Evidence</th><th>YOLO confidence</th></tr></thead><tbody>{timeline.map((item, index) => <tr key={`${item.frame_number}-${index}`}><td>{item.timestamp_sec == null ? 'Unavailable' : `${item.timestamp_sec.toFixed(1)} s`}</td><td>{item.frame_number}</td><td>{item.class_name || item.class}</td><td>{(item.confidence * 100).toFixed(1)}%</td></tr>)}</tbody></table>}
        </section>
        <section className="card"><h3>Investigation summary</h3><p>{result.investigation_summary}</p><small>{result.disclaimer}</small></section>
      </div>
    )
  }

  const detections = result.detections || []
  const score = result.risk?.overall_risk ?? 0
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
        <span className={`risk-pill ${score >= 70 ? 'high' : score >= 40 ? 'medium' : 'low'}`}>Threat level: {result.risk?.threat_level || result.risk?.risk_label || 'Low'}</span>
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
