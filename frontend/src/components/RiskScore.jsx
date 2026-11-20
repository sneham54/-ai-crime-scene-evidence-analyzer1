import React from 'react'

const LEVEL_COLORS = {
  low: '#22c55e',
  moderate: '#eab308',
  high: '#f97316',
  critical: '#ef4444',
}

function formatConfidence(value) {
  return value == null ? 'Unavailable' : `${(Number(value) * 100).toFixed(1)}% model output confidence`
}

export default function RiskScore({ risk }) {
  if (!risk) {
    return <div className="card"><h3 style={{ marginTop: 0 }}>Risk Analysis</h3><p style={{ color: 'var(--muted)' }}>No risk data available. Run an analysis first.</p></div>
  }

  const score = Number(risk.overall_risk ?? risk.impact_score ?? 0)
  const thresholds = risk.thresholds || { moderate: 25, high: 50, critical: 75 }
  const level = risk.risk_level || (score >= thresholds.critical ? 'Critical' : score >= thresholds.high ? 'High' : score >= thresholds.moderate ? 'Moderate' : 'Low')
  const color = LEVEL_COLORS[String(level).toLowerCase()] || LEVEL_COLORS.low
  const factors = risk.contributing_factors || []
  const confidence = risk.confidence_indicators || {}
  const completeness = risk.evidence_completeness || {}
  const tracking = risk.tracking_information || {}
  const events = risk.timestamped_events || []

  return (
    <div className="card risk-analysis-panel">
      <h3 style={{ marginTop: 0 }}>Investigative Risk Analysis</h3>
      <div style={{ display: 'flex', alignItems: 'center', gap: 24, marginBottom: 20 }}>
        <div style={{ width: 120, height: 120, borderRadius: '50%', border: `8px solid ${color}`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column' }}>
          <div style={{ fontSize: 32, fontWeight: 700 }}>{score}</div><div style={{ fontSize: 11, color: 'var(--muted)' }}>/ 100</div>
        </div>
        <div>
          <div style={{ fontSize: 20, fontWeight: 700, color }}>{level} risk</div>
          <div style={{ color: 'var(--muted)', fontSize: 13, marginTop: 5 }}>Thresholds: Moderate {thresholds.moderate}+ · High {thresholds.high}+ · Critical {thresholds.critical}+</div>
          <div style={{ fontSize: 13, marginTop: 7 }}>Detected object classes: {risk.detected_objects?.join(', ') || 'none recorded'}</div>
        </div>
      </div>

      <p className="muted-copy">{risk.score_explanation}</p>
      <h4>Contributing evidence</h4>
      {factors.length ? <ul>{factors.map((factor, index) => <li key={`${factor.name}-${index}`}><strong>{factor.name}</strong>: +{Number(factor.points).toFixed(1)} points · {factor.note || factor.source}</li>)}</ul> : <p className="muted-copy">No scored evidence factors were supplied.</p>}

      <div className="risk-evidence-summary">
        <p><strong>Classifier:</strong> {formatConfidence(confidence.classifier_output_confidence)}</p>
        <p><strong>Detector:</strong> {formatConfidence(confidence.detector_output_confidence)}</p>
        <p><strong>Evidence completeness:</strong> {Math.round((Number(completeness.coverage) || 0) * 100)}% of applicable inputs available</p>
        <p><strong>Tracking:</strong> {tracking.method || 'unavailable'} · {tracking.unique_track_id_count || 0} IDs · {tracking.consistent_track_ids?.length || 0} persistent IDs · match {tracking.match_status || 'not provided'}</p>
        <p><strong>Scene-change context:</strong> {risk.scene_change_candidate_count || 0} candidates; these add no risk points by themselves.</p>
      </div>

      {events.length > 0 && <div><h4>Timestamped activity candidates</h4><ul>{events.map((event, index) => <li key={`${event.category}-${event.start_timestamp_sec}-${index}`}>{event.category} · {event.start_timestamp_sec ?? 'start unavailable'}–{event.end_timestamp_sec ?? 'end unavailable'} s · {formatConfidence(event.confidence)}</li>)}</ul></div>}
      {risk.warnings?.length > 0 && <div role="status"><h4>Evidence limitations</h4><ul>{risk.warnings.map((warning, index) => <li key={`${warning}-${index}`}>{warning}</li>)}</ul></div>}
      <p style={{ color: 'var(--muted)', fontSize: 12, marginBottom: 0 }}>{risk.score_disclaimer || 'Investigative prioritization indicator only; not proof of criminal activity.'}</p>
    </div>
  )
}
