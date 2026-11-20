import React, { useState } from 'react'
import api from '../services/api'

export default function ExperimentalFeatureAnalysis({ videoFilepath }) {
  const [featureFile, setFeatureFile] = useState(null)
  const [experimentalResult, setExperimentalResult] = useState(null)
  const [experimentalError, setExperimentalError] = useState('')
  const [busy, setBusy] = useState(false)
  const [jsonReport, setJsonReport] = useState(null)

  const analyzeFeatures = async event => {
    event.preventDefault()
    if (!featureFile) return
    setBusy(true)
    setExperimentalError('')
    setExperimentalResult(null)
    setJsonReport(null)
    try {
      const input = JSON.parse(await featureFile.text())
      if (!input || typeof input !== 'object' || Array.isArray(input) || !Array.isArray(input.features)) {
        throw new Error('JSON must be an object with a top-level features array.')
      }
      if (input.features.length !== 32 || input.features.some(row =>
        !Array.isArray(row) || row.length !== 512 || row.some(value => typeof value !== 'number' || !Number.isFinite(value)))) {
        throw new Error('features must contain 32 rows of 512 finite numeric values each.')
      }
      if (input.timestamps_sec !== undefined && (
        !Array.isArray(input.timestamps_sec) || input.timestamps_sec.length !== 32 ||
        input.timestamps_sec.some((value, index, all) => typeof value !== 'number' || !Number.isFinite(value) || value < 0 || (index > 0 && value < all[index - 1]))
      )) {
        throw new Error('timestamps_sec, when provided, must contain 32 ordered finite non-negative numbers.')
      }
      const response = await api.post('/api/video/analyze-features', input)
      setExperimentalResult(response.data)
    } catch (error) {
      setExperimentalError(error.response?.data?.error || error.message || 'Experimental feature analysis failed.')
    } finally {
      setBusy(false)
    }
  }

  const analyzeUploadedVideo = async () => {
    if (!videoFilepath) return
    setBusy(true)
    setExperimentalError('')
    setExperimentalResult(null)
    setJsonReport(null)
    try {
      const response = await api.post('/api/video/analyze-experimental', { filepath: videoFilepath })
      setExperimentalResult(response.data)
    } catch (error) {
      setExperimentalError(error.response?.data?.error || error.message || 'Experimental video analysis failed.')
    } finally {
      setBusy(false)
    }
  }

  const downloadReport = async () => {
    if (!experimentalResult) return
    setBusy(true)
    setExperimentalError('')
    try {
      const response = await api.post('/api/report/generate-json', {
        file_name: featureFile?.name || 'precomputed R3D-18 features',
        experimental_anomaly: experimentalResult,
      })
      setJsonReport(response.data)
      const report = await api.get(response.data.download_url, { responseType: 'blob' })
      const url = URL.createObjectURL(new Blob([report.data], { type: 'application/json' }))
      const link = document.createElement('a')
      link.href = url
      link.download = response.data.report_name
      link.click()
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (error) {
      setExperimentalError(error.response?.data?.error || error.message || 'JSON report download failed.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="card video-timeline-card experimental-feature-card">
      <div className="panel-heading"><div><h2>Experimental R3D-18 + BiGRU/MIL review</h2><p className="muted-copy">Unverified video preprocessing and candidate checkpoint architecture. Human review required.</p></div><span className="count-pill">Experimental</span></div>
      {videoFilepath && <div className="video-action-row"><button className="secondary" type="button" onClick={analyzeUploadedVideo} disabled={busy}>{busy ? 'Extracting features and scoring…' : 'Run experimental analysis on this uploaded video'}</button></div>}
      <form onSubmit={analyzeFeatures}>
        <p className="muted-copy">Optional: choose a local JSON file with a top-level <code>features</code> array shaped (32, 512). Video uploads are processed with Kinetics-400 R3D-18 preprocessing, which may differ from the unrecovered training setup.</p>
        <div className="video-action-row">
          <input type="file" accept=".json,application/json" onChange={event => { setFeatureFile(event.target.files?.[0] || null); setExperimentalError('') }} />
          <button className="primary" type="submit" disabled={!featureFile || busy}>{busy ? 'Scoring segments…' : 'Run experimental scoring'}</button>
        </div>
        {featureFile && <small>Selected: {featureFile.name} ({(featureFile.size / 1024).toFixed(1)} KB)</small>}
      </form>
      {experimentalError && <p role="alert" className="muted-copy">{experimentalError}</p>}
      {experimentalResult && <>
        <p><strong>Unverified experimental output · Human review required · Scores are raw logits, not probabilities, accuracy measurements, or confirmed crime detections.</strong></p>
        <p className="muted-copy">Candidate architecture fidelity: {experimentalResult.candidate_architecture_fidelity}. No incident boundaries are inferred.</p>
        {experimentalResult.segment_mapping && <p className="muted-copy">{experimentalResult.segment_mapping}</p>}
        <div className="table-wrap"><table className="table"><thead><tr><th>Segment</th><th>Raw logit</th><th>Timestamp</th></tr></thead><tbody>
          {(experimentalResult.segment_scores || []).map(segment => <tr key={segment.segment_index}><td>{segment.segment_index}</td><td>{Number(segment.raw_logit).toFixed(6)}</td><td>{segment.timestamp_sec == null ? 'Not supplied' : `${Number(segment.timestamp_sec).toFixed(3)} s`}</td></tr>)}
        </tbody></table></div>
        <ul className="muted-copy">{(experimentalResult.limitations || []).map(item => <li key={item}>{item}</li>)}</ul>
        <button className="secondary" type="button" onClick={downloadReport} disabled={busy}>{busy ? 'Preparing report…' : 'Download experimental JSON report'}</button>
        {jsonReport && <small>Saved report: {jsonReport.report_name}</small>}
      </>}
    </section>
  )
}
