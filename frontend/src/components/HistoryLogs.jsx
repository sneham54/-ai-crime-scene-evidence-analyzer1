import React, { useState, useEffect } from 'react'
import api from '../services/api'

export default function HistoryLogs() {
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.get('/api/dashboard/history')
      .then(res => setLogs(res.data.logs || []))
      .catch(() => setLogs([]))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Chain of Custody / History Logs</h3>
      {loading ? (
        <p style={{ color: 'var(--muted)' }}>Loading...</p>
      ) : logs.length === 0 ? (
        <p style={{ color: 'var(--muted)' }}>
          No logs yet. (Requires MongoDB connection). Actions will be recorded after uploads and analysis.
        </p>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Action</th>
              <th>User</th>
              <th>Evidence ID</th>
              <th>Hash</th>
            </tr>
          </thead>
          <tbody>
            {logs.map((log, i) => (
              <tr key={i}>
                <td style={{ fontSize: 12 }}>{log.timestamp?.slice(0, 19)}</td>
                <td>{log.action}</td>
                <td style={{ fontSize: 12 }}>{log.user_id?.slice(0, 12)}</td>
                <td style={{ fontSize: 12 }}>{log.evidence_id?.slice(0, 12)}...</td>
                <td style={{ fontSize: 11, color: 'var(--muted)' }}>{log.integrity_hash?.slice(0, 12)}...</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
