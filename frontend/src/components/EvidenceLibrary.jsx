import { useEffect, useMemo, useState } from 'react'
import api, { BACKEND_URL } from '../services/api'

const CATEGORIES = [
  ['assault', 'Assault'],
  ['robbery', 'Robbery'],
  ['shooting', 'Shooting'],
]

function formatTimestamp(value) {
  const seconds = Number(value)
  if (!Number.isFinite(seconds)) return '—'
  const minutes = Math.floor(seconds / 60)
  return `${String(minutes).padStart(2, '0')}:${(seconds % 60).toFixed(2).padStart(5, '0')}`
}

function messageFor(error) {
  return error.response?.data?.error || error.userMessage || error.message || 'Could not load saved evidence.'
}

export default function EvidenceLibrary() {
  const [category, setCategory] = useState('assault')
  const [catalog, setCatalog] = useState([])
  const [savedReports, setSavedReports] = useState([])
  const [videos, setVideos] = useState([])
  const [eventPackages, setEventPackages] = useState([])
  const [videoId, setVideoId] = useState('')
  const [detail, setDetail] = useState(null)
  const [artifactId, setArtifactId] = useState('')
  const [mediaKind, setMediaKind] = useState('annotated')
  const [mediaUrl, setMediaUrl] = useState('')
  const [reportUrl, setReportUrl] = useState('')
  const [caseId, setCaseId] = useState('')
  const [page, setPage] = useState(0)
  const [loading, setLoading] = useState(true)
  const [savingReport, setSavingReport] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    api.get('/api/evidence/categories')
      .then(response => { if (active) setCatalog(response.data.categories || []) })
      .catch(err => { if (active) setError(messageFor(err)) })
    return () => { active = false }
  }, [])

  useEffect(() => {
    let active = true
    setLoading(true)
    setError('')
    setDetail(null)
    setArtifactId('')
    setMediaUrl('')
    setReportUrl('')
    api.get(`/api/evidence/${category}/videos`)
      .then(response => {
        if (!active) return
        const items = response.data.videos || []
        setVideos(items)
        const first = items.find(item => item.artifact_set_count > 0) || items[0]
        setVideoId(first?.video_id || '')
      })
      .catch(err => { if (active) { setVideos([]); setVideoId(''); setError(messageFor(err)) } })
      .finally(() => { if (active) setLoading(false) })
    api.get(`/api/evidence/${category}/reports`)
      .then(response => { if (active) setSavedReports(response.data.files || []) })
      .catch(() => { if (active) setSavedReports([]) })
    return () => { active = false }
  }, [category])

  useEffect(() => {
    let active = true
    api.get(`/api/evidence/event-evidence/${category}`)
      .then(response => { if (active) setEventPackages(response.data.events || []) })
      .catch(err => { if (active) { setEventPackages([]); setError(messageFor(err)) } })
    return () => { active = false }
  }, [category])

  useEffect(() => {
    if (!videoId) { setDetail(null); return undefined }
    let active = true
    setDetail(null)
    setArtifactId('')
    setMediaUrl('')
    api.get(`/api/evidence/${category}/videos/${encodeURIComponent(videoId)}`)
      .then(response => {
        if (!active) return
        const next = response.data
        setDetail(next)
        const firstSet = next.result_sets?.[0]
        setArtifactId(firstSet?.artifact_id || '')
        setMediaKind(firstSet?.annotated_available ? 'annotated' : 'source')
      })
      .catch(err => { if (active) setError(messageFor(err)) })
    return () => { active = false }
  }, [category, videoId])

  const selectedArtifact = useMemo(
    () => detail?.result_sets?.find(item => item.artifact_id === artifactId) || null,
    [detail, artifactId],
  )
  const records = selectedArtifact?.records || []
  const pageSize = 100
  const pageCount = Math.ceil(records.length / pageSize)
  const visibleRecords = records.slice(page * pageSize, (page + 1) * pageSize)

  useEffect(() => setPage(0), [artifactId, videoId, category])

  useEffect(() => {
    if (!detail || !videoId) { setMediaUrl(''); return undefined }
    const canPlay = mediaKind === 'source'
      ? detail.source_available
      : selectedArtifact?.annotated_available
    if (!canPlay) { setMediaUrl(''); return undefined }
    let active = true
    let objectUrl = ''
    const query = new URLSearchParams({ kind: mediaKind })
    if (artifactId && mediaKind === 'annotated') query.set('artifact_id', artifactId)
    api.get(`/api/evidence/${category}/videos/${encodeURIComponent(videoId)}/media?${query}`, { responseType: 'blob' })
      .then(response => {
        if (!active) return
        objectUrl = URL.createObjectURL(response.data)
        setMediaUrl(objectUrl)
      })
      .catch(err => { if (active) { setMediaUrl(''); setError(messageFor(err)) } })
    return () => {
      active = false
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [category, videoId, detail, artifactId, mediaKind, selectedArtifact])

  const createReport = async () => {
    if (!artifactId || !videoId) return
    setSavingReport(true)
    setError('')
    setReportUrl('')
    try {
      const payloadResponse = await api.get(
        `/api/evidence/${category}/videos/${encodeURIComponent(videoId)}/report-payload`,
        { params: { artifact_id: artifactId } },
      )
      const report = {
        ...payloadResponse.data.report,
        case: { ...payloadResponse.data.report.case, case_id: caseId.trim() },
      }
      const saved = await api.post('/api/report/generate-json', report)
      setReportUrl(`${BACKEND_URL}${saved.data.download_url}`)
    } catch (err) {
      setError(messageFor(err))
    } finally {
      setSavingReport(false)
    }
  }

  const downloadSavedReport = async report => {
    try {
      const response = await api.get(report.download_url, { responseType: 'blob' })
      const url = URL.createObjectURL(response.data)
      const link = document.createElement('a')
      link.href = url
      link.download = report.name
      link.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(messageFor(err))
    }
  }

  const summary = catalog.find(item => item.category === category)
  const isActivity = selectedArtifact?.result_type === 'activity_prediction'
  const bboxText = bbox => bbox ? `${bbox.x1}, ${bbox.y1}, ${bbox.x2}, ${bbox.y2}` : '—'

  return (
    <div className="evidence-library">
      <section className="card evidence-library-controls">
        <label>
          <span className="evidence-label">Category</span>
          <select value={category} onChange={event => setCategory(event.target.value)}>
            {CATEGORIES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        <label>
          <span className="evidence-label">Video</span>
          <select value={videoId} onChange={event => setVideoId(event.target.value)} disabled={loading || !videos.length}>
            {!videos.length && <option value="">No source videos or saved results found</option>}
            {videos.map(video => (
              <option key={video.video_id} value={video.video_id}>
                {video.video_name} · {video.artifact_set_count ? `${video.record_count} saved records` : video.result_status}
              </option>
            ))}
          </select>
        </label>
      </section>

      {error && <div className="card evidence-error" role="alert">{error}</div>}
      {loading && <div className="card">Loading saved evidence…</div>}

      {!loading && summary && (
        <>
          <section className="evidence-summary-grid">
            <article className="card evidence-summary-card"><span>Source videos</span><strong>{summary.source_video_count}</strong></article>
            <article className="card evidence-summary-card"><span>Videos with saved results</span><strong>{summary.videos_with_results}</strong></article>
            <article className="card evidence-summary-card"><span>Status records</span><strong>{summary.status_record_count}</strong></article>
            <article className="card evidence-summary-card"><span>Saved result records</span><strong>{summary.result_record_count}</strong></article>
          </section>
          <p className="muted-copy evidence-method-note">Pipeline type: {summary.method}. Object detections and activity predictions are model outputs, not findings of guilt.</p>
        </>
      )}

      {!loading && detail && (
        <div className="evidence-library-grid">
          <section className="card evidence-player-card">
            <div className="panel-heading">
              <div><h2>{detail.video_name}</h2><p className="panel-note">Processing status: {detail.result_status}</p></div>
              <span className="panel-note">{detail.source_available ? 'Source available' : 'Source unavailable'}</span>
            </div>
            {detail.result_sets?.length > 0 && (
              <label className="evidence-run-select">
                Saved result set
                <select value={artifactId} onChange={event => setArtifactId(event.target.value)}>
                  {detail.result_sets.map(item => <option key={item.artifact_id} value={item.artifact_id}>{item.name} · {item.records.length} records</option>)}
                </select>
              </label>
            )}
            {(detail.source_available || selectedArtifact?.annotated_available) && (
              <div className="evidence-media-switch">
                {detail.source_available && <button className={mediaKind === 'source' ? 'tab-button active' : 'tab-button'} onClick={() => setMediaKind('source')}>Source video</button>}
                {selectedArtifact?.annotated_available && <button className={mediaKind === 'annotated' ? 'tab-button active' : 'tab-button'} onClick={() => setMediaKind('annotated')}>Annotated video</button>}
              </div>
            )}
            {mediaUrl ? <video className="evidence-video" src={mediaUrl} controls preload="metadata" /> : <div className="image-placeholder">{detail.source_available || selectedArtifact?.annotated_available ? 'Loading video…' : 'No source or annotated video is available for this result.'}</div>}
            {detail.model_provenance && <p className="panel-note">{detail.model_provenance}</p>}
          </section>

          <section className="card evidence-results-card">
            <div className="panel-heading"><div><h2>{isActivity ? 'Activity predictions' : 'Object detections'}</h2><p className="panel-note">Saved records: {records.length} · not ground-truth confirmation</p></div></div>
            {detail.result_sets?.length > 0 ? (
              <>
                <div className="evidence-table-wrap">
                  <table className="evidence-table">
                    <thead><tr>
                      <th>Timestamp</th><th>Frame</th><th>{isActivity ? 'Prediction' : 'Class'}</th>
                      <th>Confidence</th>{!isActivity && <><th>Track ID</th><th>Bounding box</th></>}
                    </tr></thead>
                    <tbody>
                      {visibleRecords.map((item, index) => <tr key={`${item.frame}-${item.timestamp_seconds}-${item.class_name}-${page}-${index}`}>
                        <td>{formatTimestamp(item.timestamp_seconds)}</td>
                        <td>{item.frame ?? '—'}</td>
                        <td>{item.class_name || '—'}</td>
                        <td>{item.confidence == null ? '—' : `${(item.confidence * 100).toFixed(1)}%`}</td>
                        {!isActivity && <><td>{item.track_id || '—'}</td><td>{bboxText(item.bbox)}</td></>}
                      </tr>)}
                    </tbody>
                  </table>
                </div>
                {pageCount > 1 && <div className="evidence-pagination"><button className="secondary" disabled={page === 0} onClick={() => setPage(value => Math.max(0, value - 1))}>Previous</button><span>Showing {page * pageSize + 1}–{Math.min((page + 1) * pageSize, records.length)} of {records.length}</span><button className="secondary" disabled={page + 1 >= pageCount} onClick={() => setPage(value => Math.min(pageCount - 1, value + 1))}>Next</button></div>}
                {selectedArtifact.events?.length > 0 && <div className="evidence-event-candidates">
                  <h3>Candidate intervals · human review required</h3>
                  {selectedArtifact.events.map((event, index) => <article key={`${event.start_timestamp_seconds}-${index}`}>
                    <strong>{event.review_label || 'Unverified candidate'}</strong>
                    <span>{formatTimestamp(event.start_timestamp_seconds)}–{formatTimestamp(event.end_timestamp_seconds)}</span>
                    {event.evidence_summary && <small>{event.evidence_summary}</small>}
                    {event.note && <small>{event.note}</small>}
                  </article>)}
                </div>}
                <div className="evidence-report-actions">
                  <label className="evidence-case-id">Case ID (optional)<input value={caseId} onChange={event => setCaseId(event.target.value)} placeholder="Enter an existing case reference" /></label>
                  <button className="primary" onClick={createReport} disabled={savingReport}>{savingReport ? 'Preparing report…' : 'Add this result set to an investigation report'}</button>
                  {reportUrl && <a href={reportUrl}>Download saved JSON report</a>}
                </div>
              </>
            ) : (
              <p className="muted-copy">No saved prediction file is linked to this video. This view does not run inference.</p>
            )}
          </section>
        </div>
      )}

      {eventPackages.length > 0 && <section className="card evidence-saved-reports">
        <div className="panel-heading"><div><h2>Extracted {CATEGORIES.find(([key]) => key === category)?.[1]} event evidence</h2><p className="panel-note">Saved event packages · verification status is retained from review</p></div><span className="count-pill">{eventPackages.length}</span></div>
        <div className="object-timeline-list">{eventPackages.map(event => {
          const routeCategory = String(event.category || category).toLowerCase()
          const base = routeCategory === 'shooting'
            ? `/api/video/evidence/${encodeURIComponent(event.analysis_id)}/shooting/${encodeURIComponent(event.evidence_id)}`
            : `/api/video/evidence/${encodeURIComponent(event.analysis_id)}/events/${routeCategory}/${encodeURIComponent(event.evidence_id)}`
          return <article className="object-timeline-event" key={event.evidence_id}>
            <h3>{event.category} · {event.original_video?.split(/[\\/]/).pop()}</h3>
            <p>{formatTimestamp(event.start_timestamp_sec)}–{formatTimestamp(event.end_timestamp_sec)} · classifier {event.classifier_confidence == null ? 'Unavailable' : `${(event.classifier_confidence * 100).toFixed(1)}%`} · {event.verification_status}</p>
            <p>{event.detections?.length || 0} detections · {event.tracking_groups?.length || 0} tracked objects · source SHA-256 {event.original_video_sha256}</p>
            <div className="video-export-links">
              <a className="secondary" href={`${BACKEND_URL}${base}/original_clip.mp4`} target="_blank" rel="noreferrer">Original event clip</a>
              {event.artifacts?.annotated_clip && <a className="secondary" href={`${BACKEND_URL}${base}/annotated_clip.mp4`} target="_blank" rel="noreferrer">Annotated event clip</a>}
              <a className="secondary" href={`${BACKEND_URL}${base}/detections.csv`} download>Detection CSV</a>
              <a className="secondary" href={`${BACKEND_URL}${base}/event.json`} download>Event JSON</a>
              {event.artifacts?.integrity_manifest && <a className="secondary" href={`${BACKEND_URL}${base}/event.integrity.json`} download>Integrity manifest</a>}
            </div>
            {event.artifacts?.keyframes?.map(path => {
              const name = path.split(/[\\/]/).pop()
              return <img className="evidence-keyframe" key={path} src={`${BACKEND_URL}${base}/keyframes/${encodeURIComponent(name)}`} alt={`Original CCTV keyframe ${name}`} loading="lazy" />
            })}
            {event.artifacts?.annotated_keyframes?.map(path => {
              const name = path.split(/[\\/]/).pop()
              return <img className="evidence-keyframe" key={path} src={`${BACKEND_URL}${base}/annotated_keyframes/${encodeURIComponent(name)}`} alt={`Detection annotated keyframe ${name}`} loading="lazy" />
            })}
            {event.detections?.length > 0 && <details><summary>Detection records</summary><div className="detection-observation-list">{event.detections.map((item, index) => <div className="detection-observation-row" key={`${item.frame_number}-${index}`}>
              <span>{formatTimestamp(item.timestamp_seconds)} · frame {item.frame_number}</span><span>{item.class_name} · {item.confidence == null ? 'Unavailable' : `${(item.confidence * 100).toFixed(1)}%`}</span><span>Track {item.track_id || 'Unavailable'} · {bboxText(item.bbox)}</span>
            </div>)}</div></details>}
          </article>
        })}</div>
      </section>}

      {!loading && savedReports.length > 0 && (
        <section className="card evidence-saved-reports">
          <h2>Saved reports and audits</h2>
          <div className="evidence-report-list">
            {savedReports.map(report => <div className="evidence-report-file" key={report.report_id}>
              <div><strong>{report.name}</strong><small>{Math.max(1, Math.round(report.size_bytes / 1024))} KB</small></div>
              <button className="secondary" onClick={() => downloadSavedReport(report)}>Download</button>
            </div>)}
          </div>
        </section>
      )}
    </div>
  )
}
