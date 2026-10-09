import React from 'react'

export default function RiskScore({ risk }) {
  if (!risk) {
    return (
      <div className="card">
        <h3 style={{ marginTop: 0 }}>Risk Analysis</h3>
        <p style={{ color: 'var(--muted)' }}>No risk data available. Run an analysis first.</p>
      </div>
    )
  }

  const score = risk.overall_risk || 0
  const color = score >= 70 ? '#ef4444' : score >= 40 ? '#f59e0b' : '#22c55e'

  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Risk Analysis</h3>

      <div style={{ display: 'flex', alignItems: 'center', gap: 24, marginBottom: 24 }}>
        <div style={{
          width: 120, height: 120, borderRadius: '50%',
          border: `8px solid ${color}`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexDirection: 'column'
        }}>
          <div style={{ fontSize: 32, fontWeight: 700 }}>{score}</div>
          <div style={{ fontSize: 11, color: 'var(--muted)' }}>/ 100</div>
        </div>
        <div>
          <div style={{ fontSize: 20, fontWeight: 700, color }}>{risk.risk_label}</div>
          <div style={{ color: 'var(--muted)', fontSize: 14, marginTop: 4 }}>
            Threat: {risk.threat_level} | Severity: {risk.severity}
          </div>
          <div style={{ fontSize: 13, marginTop: 6 }}>
            Probability: {(risk.probability * 100).toFixed(0)}% | Weapons: {risk.weapon_count || 0}
          </div>
        </div>
      </div>

      {risk.reasons && risk.reasons.length > 0 && (
        <div>
          <h4 style={{ margin: '0 0 8px', fontSize: 14 }}>Reasons</h4>
          <ul style={{ margin: 0, paddingLeft: 18, color: 'var(--muted)', fontSize: 13 }}>
            {risk.reasons.map((r, i) => <li key={i}>{r}</li>)}
          </ul>
        </div>
      )}
    </div>
  )
}
