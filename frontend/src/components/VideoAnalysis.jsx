import React, { useRef, useState } from 'react'
import api from '../services/api'
import { BACKEND_URL } from '../services/api'
import { normalizeVideoEvents } from '../utils/videoEvents'

function formatTimestamp(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return 'Unavailable'
  const value = Math.max(0, Number(seconds))
  const minutes = Math.floor(value / 60)
  return `${String(minutes).padStart(2, '0')}:${(value - minutes * 60).toFixed(2).padStart(5, '0')}`
}

function formatConfidence(value) {
  return value == null || !Number.isFinite(Number(value)) ? 'Unavailable' : `${(Number(value) * 100).toFixed(1)}%`
}

function Probabilities({ values, title }) {
  if (!values || !Object.keys(values).length) return null
  return <div className="classifier-probability-block">
    {title && <h3>{title}</h3>}
    <div className="classification-probabilities">{Object.entries(values).map(([name, value]) => (
      <div className="probability-row" key={name}>
        <span>{name}</span><div><i style={{ width: `${Math.max(0, Math.min(100, Number(value) * 100))}%` }} /></div>
        <b>{formatConfidence(value)}</b>
      </div>
    ))}</div>
  </div>
}

export default function VideoAnalysis({ result, onUpload, onReport }) {
  const [eventEvidence, setEventEvidence] = useState([])
  const [creatingEvidence, setCreatingEvidence] = useState(null)
  const [evidenceError, setEvidenceError] = useState('')
  const videoInfo = result.video_info || {}
  const classification = result.classification || result.event_classification || {}
  const tracking = result.tracking || {}
  const trackingMatch = result.tracking_data_match || {}
  const frames = result.frame_results || []
  const detections = result.detections || frames.flatMap(frame => frame.detections || [])
  const objectEvents = result.evidence_events || []
  const activityEvents = normalizeVideoEvents(result)
  const thumbnails = new Map((result.frame_thumbnails || []).map(item => [item.frame_number, item.image]))
  const classifierCoverage = videoInfo.classifier_sampling_coverage || classification.sampling_coverage
  const playerRef = useRef(null)
  const seekToTimestamp = seconds => {
    if (playerRef.current && seconds != null) {
      playerRef.current.currentTime = Number(seconds)
      playerRef.current.play?.()
    }
  }
  const isSavedArtifact = ['saved_tracking_csv', 'saved_txt_labels'].includes(tracking.method)
  const verifiedMatch = ['EXACT_MATCH', 'METADATA_MATCH'].includes(trackingMatch.status)
  const trackingIds = trackingMatch.tracking_id_count ?? tracking.tracking_id_count
  const statusLabel = classification.classification_status || (classification.label === 'Unavailable' ? 'unavailable' : 'provisional')
  const createEventEvidence = async event => {
    if (!result.analysis_id || creatingEvidence != null) return
    const key = `${event.start_timestamp_sec}-${event.end_timestamp_sec}`
    setCreatingEvidence(key)
    setEvidenceError('')
    try {
      const category = String(result.category || '').toLowerCase()
      const response = await api.post(`/api/video/event-evidence/${encodeURIComponent(result.analysis_id)}`, {
        start_timestamp_sec: event.start_timestamp_sec,
        end_timestamp_sec: event.end_timestamp_sec,
        confidence: event.confidence,
        verification_status: event.verification_status || result.verification_status || 'unverified',
        classifier_metadata: classification,
      })
      const item = response.data.event_evidence || response.data.shooting_event
      setEventEvidence(items => [...items.filter(existing => existing.evidence_id !== item.evidence_id), item])
    } catch (error) {
      setEvidenceError(error.response?.data?.error || error.message || 'Could not extract event evidence.')
    } finally {
      setCreatingEvidence(null)
    }
  }

  return <div className="analysis-workspace video-analysis-workspace">
    <section className="card video-source-card">
      <div className="section-kicker">Original uploaded video</div>
      <h2>{result.file_name || 'Uploaded video'}</h2>
      {result.media_preview
        ? <video ref={playerRef} className="video-source-preview" src={result.media_preview} controls playsInline />
        : <div className="video-source-placeholder">Original video preview unavailable</div>}
      <div className="video-action-row">
        <button className="primary" onClick={onUpload}>+ Analyze another video</button>
        <button className="secondary" onClick={onReport}>Create PDF report</button>
      </div>
      <p className="video-processing-note">{result.file_name || 'Filename unavailable'} · {result.category || 'Category unavailable'} · {videoInfo.duration_sec == null ? 'Duration unavailable' : `${Number(videoInfo.duration_sec).toFixed(2)} s`} · {videoInfo.fps == null ? 'FPS unavailable' : `${Number(videoInfo.fps).toFixed(3)} FPS`} · {videoInfo.width && videoInfo.height ? `${videoInfo.width} × ${videoInfo.height}` : 'Resolution unavailable'} · {videoInfo.total_frames ?? 'Unknown'} source frames</p>
      <p className="video-processing-note">Processing: {videoInfo.processing_status || 'status unavailable'}</p>
      {videoInfo.timestamp_warning && <p className="muted-copy">Timestamp mapping: {videoInfo.timestamp_warning}</p>}
    </section>

    <section className="card video-tracking-card">
      <div className="panel-heading"><div><div className="section-kicker">Evidence source</div><h2>Tracking information</h2></div><span className={`count-pill ${verifiedMatch ? 'matched' : ''}`}>{trackingMatch.status || 'NOT_CHECKED'}</span></div>
      <p>{trackingMatch.reason || (verifiedMatch ? 'Existing tracking artifact was matched.' : 'No verified saved tracking match was reported.')}</p>
      <div className="tracking-summary-grid">
        <span>Artifact: <b>{trackingMatch.artifact_type || tracking.method || 'Unavailable'}</b></span>
        <span>Detections: <b>{trackingMatch.detection_count ?? tracking.detection_count ?? detections.length}</b></span>
        <span>Rows with IDs: <b>{trackingMatch.tracked_record_count ?? tracking.tracked_detection_count ?? detections.filter(item => item.track_id != null).length}</b></span>
        <span>Unique IDs: <b>{trackingIds ?? 'Unavailable'}</b></span>
        <span>Frame coverage: <b>{trackingMatch.frame_coverage_count ?? videoInfo.frames_analyzed ?? frames.length} frames</b></span>
        <span>Artifact file: <b>{trackingMatch.artifact_filename || trackingMatch.source_filename || 'Unavailable'}</b></span>
      </div>
      {trackingMatch.metadata_verification && <p className="muted-copy">Metadata verification: {Object.entries(trackingMatch.metadata_verification).map(([key, value]) => `${key}: ${value ? 'verified' : 'unavailable'}`).join(' · ')}</p>}
      <p className="muted-copy">{verifiedMatch && isSavedArtifact ? 'Existing artifact detections were reused; YOLO/ByteTrack was not rerun.' : `Tracking fallback: ${tracking.method || 'not available'}. ${tracking.used ? 'IDs shown below come from this analysis.' : 'Tracking IDs are unavailable; detection boxes do not imply identities.'}`}</p>
      {tracking.annotated_video_required === false && <p className="muted-copy">The original uploaded video remains the player source. An annotated video is not required.</p>}
    </section>

    <section className="card video-classification-card">
      <div className="panel-heading">
        <div><div className="section-kicker">Whole-video primary prediction · {classification.classifier_backend === 'experimental_densenet3d_attention_biconvlstm' ? 'experimental DenseNet-3D + attention + bidirectional ConvLSTM' : 'existing ResNet18 + LSTM'}</div><h2>{classification.label || 'Classification unavailable'}</h2></div>
        <span className="count-pill">{formatConfidence(classification.confidence)} model confidence</span>
      </div>
      <p className="classification-status-note"><b>Status: {statusLabel.replaceAll('_', ' ')}</b> · This is provisional, unvalidated model evidence.</p>
      {classification.alternative_predictions?.length > 0 && <div className="classification-alternatives"><b>Conflicting temporal alternative:</b> {classification.alternative_predictions.map(item => `${item.label} (${formatConfidence(item.confidence)})`).join(', ')}. Review the original video; these are not separate incident findings.</div>}
      <Probabilities values={classification.probabilities} title="Whole-video class probabilities" />
      {classification.temporal_prediction && <>
        <p className="muted-copy">Temporal-window aggregate: {classification.temporal_prediction.label || classification.temporal_prediction.top_class || 'Unavailable'} · {formatConfidence(classification.temporal_prediction.confidence)}. Window scores are shown separately and are not calibrated accuracy.</p>
        <Probabilities values={classification.temporal_prediction.probabilities} title="Temporal-window class probabilities" />
      </>}
      {classification.classifier_note && <p className="muted-copy">{classification.classifier_note}</p>}
      {classification.model_checkpoint && <p className="muted-copy">Checkpoint: {classification.model_checkpoint}</p>}
      <p className="muted-copy">A Normal prediction does not establish that no incident occurred. Model confidence is not measured accuracy.</p>
    </section>

    <section className="card video-timeline-card">
      <div className="panel-heading"><div><div className="section-kicker">Unverified candidate locations</div><h2>Activity timeline</h2></div><span className="count-pill">{activityEvents.length} candidate intervals</span></div>
      <p className="muted-copy">Activity intervals are model generated candidates. Boundaries are unvalidated and require review against the original video.</p>
      {activityEvents.length ? <div className="activity-event-list">{activityEvents.map((event, index) => (
        <article className="activity-event-row" key={`${event.category}-${event.start_timestamp_sec}-${index}`}>
          <strong>{event.category} candidate</strong>
          <span>Window confidence: {formatConfidence(event.confidence)} · {event.window_count || 1} window(s) · {event.verification_status || 'unverified_candidate'}</span>
          <div className="activity-event-times">
            <button type="button" className="text-button" onClick={() => seekToTimestamp(event.start_timestamp_sec)}>Seek to start · {formatTimestamp(event.start_timestamp_sec)}</button>
            <button type="button" className="text-button" onClick={() => seekToTimestamp(event.end_timestamp_sec)}>Seek to end · {formatTimestamp(event.end_timestamp_sec)}</button>
          </div>
          <span>Supporting detections: {event.supporting_detection_count ?? 0} · Objects: {(event.detected_objects || []).join(', ') || 'None recorded'} · Track IDs: {(event.tracking_ids || []).join(', ') || 'None available'}</span>
          {event.keyframes?.map((keyframe, keyframeIndex) => <div className="event-keyframe-preview" key={`${keyframe.frame_index}-${keyframeIndex}`}>
            {keyframe.url && <img className="evidence-keyframe" src={`${BACKEND_URL}${keyframe.url}`} alt={`${event.category} candidate keyframe ${keyframe.frame_index}`} loading="lazy" />}
            <button type="button" className="text-button" onClick={() => seekToTimestamp(keyframe.timestamp_sec)}>Review keyframe · frame {keyframe.frame_index} · {formatTimestamp(keyframe.timestamp_sec)}</button>
          </div>)}
          {event.window_predictions?.length > 0 && <details><summary>Individual window predictions</summary>{event.window_predictions.map((window, windowIndex) => <div className="detection-observation-row" key={`${window.start_timestamp_sec}-${windowIndex}`}>
            <span>{formatTimestamp(window.start_timestamp_sec)}–{formatTimestamp(window.end_timestamp_sec)}</span>
            <span>{window.label}{window.raw_label !== window.label ? ` (raw: ${window.raw_label})` : ''} · {formatConfidence(window.confidence)}</span>
            <span>Shooting probability: {formatConfidence(window.probabilities?.Shooting)}</span>
          </div>)}</details>}
          {['shooting', 'assault', 'robbery'].includes(result.category?.toLowerCase()) && <button type="button" className="secondary" disabled={creatingEvidence != null} onClick={() => createEventEvidence(event)}>{creatingEvidence === `${event.start_timestamp_sec}-${event.end_timestamp_sec}` ? 'Extracting evidence…' : `Create ${result.category} evidence package`}</button>}
        </article>
      ))}</div> : <p className="muted-copy">No timestamped candidate interval is available for this prediction. This does not establish that no incident occurred.</p>}
      {evidenceError && <p className="evidence-error" role="alert">{evidenceError}</p>}
      {eventEvidence.map(item => {
        const category = String(item.category || result.category || 'shooting').toLowerCase()
        const artifactBase = category === 'shooting'
          ? `/api/video/evidence/${result.analysis_id}/shooting/${item.evidence_id}`
          : `/api/video/evidence/${result.analysis_id}/events/${category}/${item.evidence_id}`
        return <article className="activity-event-row" key={item.evidence_id}>
        <strong>{item.category} evidence · {formatTimestamp(item.start_timestamp_sec)}–{formatTimestamp(item.end_timestamp_sec)}</strong>
        <span>Classifier candidate: {formatTimestamp(item.original_candidate_start_sec)}–{formatTimestamp(item.original_candidate_end_sec)} · Refined interval: {formatTimestamp(item.refined_start_sec)}–{formatTimestamp(item.refined_end_sec)} · {item.refinement_status || 'not refined'}</span>
        <span>Temporal evidence score: {formatConfidence(item.evidence_score)} · supporting detections: {item.supporting_detection_count ?? item.detections.length} · {item.verification_status}</span>
        <span>Status: {item.verification_status} · {item.detections.length} detections · {item.tracking_groups.length} tracking IDs · source SHA-256 {item.original_video_sha256}</span>
        {item.detections.length > 0 && <details><summary>Detection records</summary>{item.detections.map((detection, index) => <div className="detection-observation-row" key={`${detection.frame_number}-${index}`}>
          <span>{formatTimestamp(detection.timestamp_seconds)} · frame {detection.frame_number}</span><span>{detection.class_name} · {formatConfidence(detection.confidence)}</span><span>Track ID {detection.track_id ?? 'Unavailable'} · bbox {(detection.bbox || []).map(value => Number(value).toFixed(1)).join(', ') || 'Unavailable'}</span>
        </div>)}</details>}
        <div className="video-export-links">
          <a className="secondary" href={`${BACKEND_URL}${artifactBase}/original_clip.mp4`} target="_blank" rel="noreferrer">Original event clip</a>
          {item.artifacts.annotated_clip && <a className="secondary" href={`${BACKEND_URL}${artifactBase}/annotated_clip.mp4`} target="_blank" rel="noreferrer">Aligned tracking clip</a>}
          <a className="secondary" href={`${BACKEND_URL}${artifactBase}/detections.csv`} download>Event detections CSV</a>
          <a className="secondary" href={`${BACKEND_URL}${artifactBase}/event.json`} download>Event JSON</a>
          {item.artifacts.integrity_manifest && <a className="secondary" href={`${BACKEND_URL}${artifactBase}/event.integrity.json`} download>Event integrity manifest</a>}
        </div>
        {item.artifacts.keyframes.map(path => <img key={path} className="evidence-keyframe" src={`${BACKEND_URL}${artifactBase}/keyframes/${path.split(/[\\/]/).pop()}`} alt={`Original keyframe ${path.split(/[\\/]/).pop()}`} loading="lazy" />)}
        {item.artifacts.annotated_keyframes.map(path => <img key={path} className="evidence-keyframe" src={`${BACKEND_URL}${artifactBase}/annotated_keyframes/${path.split(/[\\/]/).pop()}`} alt={`Detection annotated keyframe ${path.split(/[\\/]/).pop()}`} loading="lazy" />)}
      </article>})}
    </section>

    <section className="card video-timeline-card">
      <div className="panel-heading"><div><div className="section-kicker">Detection evidence · not incident findings</div><h2>Detected Object Timeline</h2></div><span className="count-pill">{detections.length} detections · {objectEvents.length} grouped objects</span></div>
      <p className="muted-copy">Each row preserves its source timestamp, frame, detector confidence, bounding box and tracking ID when supplied. Preview seeks the original video player.</p>
      <p className="video-export-links">
        {result.evidence_timeline_csv && <a className="secondary" href={`${BACKEND_URL}${result.evidence_timeline_csv}`} download>Download detection CSV</a>}
        {result.evidence_events_csv && <a className="secondary" href={`${BACKEND_URL}${result.evidence_events_csv}`} download>Download grouped object CSV</a>}
        {result.analysis_json && <a className="secondary" href={`${BACKEND_URL}${result.analysis_json}`} download>Download analysis JSON</a>}
      </p>
      {objectEvents.length ? <div className="object-timeline-list">{objectEvents.map((event, index) => {
        const observations = event.observations || []
        const confidence = event.confidence_summary || {}
        return <article className="object-timeline-event" key={`${event.class}-${event.track_id ?? index}-${index}`}>
          <div className="object-event-heading">
            {thumbnails.has(event.representative_frame_number) && <img src={thumbnails.get(event.representative_frame_number)} alt={`Original frame ${event.representative_frame_number}`} />}
            <div><h3>{event.class_name || event.class || 'Unknown object'}{event.track_id != null ? ` · Track ${event.track_id}` : ' · No track ID'}</h3>
              <p>{formatTimestamp(event.start_timestamp_sec)}–{formatTimestamp(event.end_timestamp_sec)} · {event.observation_count ?? observations.length} observations</p>
              <p>Confidence min / mean / max: {formatConfidence(confidence.min)} / {formatConfidence(confidence.mean)} / {formatConfidence(confidence.max)}</p>
            </div>
            <button type="button" className="secondary" onClick={() => seekToTimestamp(event.start_timestamp_sec)}>Preview at start</button>
          </div>
          <details><summary>Show all {observations.length} detection observations</summary>
            <div className="detection-observation-list">{observations.map((item, observationIndex) => {
              const bbox = item.bbox || []
              const bboxText = bbox.length === 4 ? bbox.map(value => Number.isFinite(Number(value)) ? Number(value).toFixed(1) : 'Unavailable').join(', ') : 'Unavailable'
              return <div className="detection-observation-row" key={`${item.frame_number}-${observationIndex}`}>
                <span>{formatTimestamp(item.timestamp_sec)} · frame {item.frame_number ?? 'Unavailable'}</span>
                <span>{item.class_name || item.class || event.class || 'Unknown'} · confidence {formatConfidence(item.confidence)}</span>
                <span>Track ID: {item.track_id ?? 'Unavailable'} · bbox (x1,y1,x2,y2): {bboxText}</span>
                <button type="button" className="text-button" onClick={() => seekToTimestamp(item.timestamp_sec)}>Preview this detection</button>
              </div>
            })}</div>
          </details>
          <span className="muted-copy">Grouped detection evidence is unverified and does not confirm an incident.</span>
        </article>
      })}</div> : <div className="overview-empty">No detections were recorded in the analyzed frames.</div>}
    </section>

    <section className="card video-summary-card"><div className="section-kicker">Investigation summary</div><p>{result.investigation_summary || 'Summary unavailable.'}</p><strong>Classifier predictions, activity intervals and object detections are provisional evidence requiring human review.</strong><br /><small>{result.disclaimer}</small></section>
  </div>
}
