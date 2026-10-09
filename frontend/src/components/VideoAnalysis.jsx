import React from 'react'

function formatTimestamp(seconds) {
  const value = Math.max(0, Math.floor(Number(seconds) || 0))
  return `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(value % 60).padStart(2, '0')}`
}

export default function VideoAnalysis({ result, onUpload, onReport }) {
  const classification = result.classification || result.event_classification
  const videoInfo = result.video_info || {}
  const frames = result.frame_results || []
  const detections = result.detections || frames.flatMap(frame => frame.detections || [])
  const keyFrames = frames.length > 6
    ? Array.from({ length: 6 }, (_, index) => frames[Math.round(index * (frames.length - 1) / 5)])
    : frames
  const thumbnails = new Map((result.frame_thumbnails || []).map(item => [item.frame_number, item.image]))
  const classCount = predicate => detections.filter(predicate).length
  const personCount = classCount(item => item.category === 'Persons' || /person/i.test(item.class || ''))
  const vehicleCount = classCount(item => item.category === 'Vehicles' || /vehicle|car|truck|bus/i.test(item.class || ''))
  const weaponCount = classCount(item => item.category === 'Weapons' || /weapon|gun|knife|rifle/i.test(item.class || ''))
  const probabilities = classification?.probabilities || {}

  return (
    <div className="analysis-workspace video-analysis-workspace">
      <section className="card video-source-card">
        {result.media_preview
          ? <video className="video-source-preview" src={result.media_preview} controls />
          : <div className="video-source-placeholder">Video evidence preview unavailable</div>}
        <div className="video-action-row">
          <button className="primary" onClick={onUpload}>+ Analyze another video</button>
          <button className="secondary" onClick={onReport}>Create PDF report</button>
          <span>{result.file_name}</span>
        </div>
        <p className="video-processing-note">Processing complete · {videoInfo.total_frames || 0} source frames · {videoInfo.frames_analyzed || 0} sampled</p>
        <div className="processing-progress"><i /></div>
      </section>

      <section className="video-metrics-grid">
        {[
          ['Persons detected', personCount],
          ['Vehicles detected', vehicleCount],
          ['Weapons detected', weaponCount],
          ['Frames analyzed', videoInfo.frames_analyzed ?? frames.length],
        ].map(([label, value]) => <article className="card video-metric" key={label}><span>{label}</span><strong>{value}</strong></article>)}
      </section>

      <section className="card video-classification-card">
        <div className="panel-heading">
          <div><div className="section-kicker">Whole-video event classification · trained model</div><h2>{classification?.label || 'Classification unavailable'}</h2></div>
          <span className="count-pill">{classification?.confidence == null ? 'Review required' : `${(classification.confidence * 100).toFixed(1)}% confidence`}</span>
        </div>
        {Object.keys(probabilities).length > 0 && <div className="classification-probabilities">{Object.entries(probabilities).map(([name, value]) => <div className="probability-row" key={name}><span>{name}</span><div><i style={{ width: `${Math.max(0, Math.min(100, Number(value) * 100))}%` }} /></div><b>{(Number(value) * 100).toFixed(1)}%</b></div>)}</div>}
        {classification?.classifier_note && <p className="muted-copy">{classification.classifier_note}</p>}
      </section>

      <section className="card video-timeline-card">
        <div className="panel-heading"><div><h2>Extracted frame timeline</h2><p className="muted-copy">YOLO detections from sampled video frames</p></div><span className="count-pill">{keyFrames.length} key frames</span></div>
        {!videoInfo.timestamps_available && <p className="muted-copy">Video timing metadata was unavailable; timestamps cannot be displayed reliably.</p>}
        {keyFrames.length ? <div className="frame-timeline">{keyFrames.map((frame, index) => {
          const detections = frame.detections || []
          const timestamp = frame.timestamp_sec == null ? `Frame ${frame.frame_number}` : formatTimestamp(frame.timestamp_sec)
          return <article className="frame-card" key={`${frame.frame_number}-${index}`}>
            <div className="frame-card-art" style={thumbnails.has(frame.frame_number) ? { backgroundImage: `linear-gradient(180deg, rgba(9, 18, 34, .08), rgba(9, 18, 34, .24)), url(${thumbnails.get(frame.frame_number)})` } : undefined}>
              <span>FRAME {String(index + 1).padStart(2, '0')}</span><b>{detections.length}</b>
              <div className="frame-detection-boxes">{detections.slice(0, 4).map((item, boxIndex) => {
                const [x1, y1, x2, y2] = item.bbox || []
                if (![x1, y1, x2, y2].every(Number.isFinite) || !frame.frame_width || !frame.frame_height) return null
                return <i key={`${item.class}-${boxIndex}`} style={{ left: `${(x1 / frame.frame_width) * 100}%`, top: `${(y1 / frame.frame_height) * 100}%`, width: `${((x2 - x1) / frame.frame_width) * 100}%`, height: `${((y2 - y1) / frame.frame_height) * 100}%` }} />
              })}</div>
            </div>
            <div className="frame-card-copy"><strong>{timestamp}</strong><small>{detections.length ? detections.map(item => item.class).join(', ') : 'No objects detected'}</small></div>
          </article>
        })}</div> : <div className="overview-empty">No sampled frames are available for this video.</div>}
      </section>

      <section className="card video-summary-card"><div className="section-kicker">Investigation summary</div><p>{result.investigation_summary}</p><small>{result.disclaimer}</small></section>
    </div>
  )
}
