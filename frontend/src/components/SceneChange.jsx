import React, { useState } from 'react'
import api, { BACKEND_URL } from '../services/api'

/**
 * Scene Change Detection UI
 * Upload two views of a scene → SSIM + alignment + changed regions (synopsis obj. 6)
 */
export default function SceneChange() {
  const [fileA, setFileA] = useState(null)
  const [fileB, setFileB] = useState(null)
  const [previewA, setPreviewA] = useState(null)
  const [previewB, setPreviewB] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const onPick = (which, e) => {
    const f = e.target.files?.[0]
    if (!f) return
    const url = URL.createObjectURL(f)
    if (which === 'a') {
      setFileA(f)
      setPreviewA(url)
    } else {
      setFileB(f)
      setPreviewB(url)
    }
  }

  const compare = async () => {
    if (!fileA || !fileB) {
      setError('Please select both images (View A and View B).')
      return
    }
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const form = new FormData()
      form.append('file_a', fileA)
      form.append('file_b', fileB)
      const res = await api.post('/api/scene-change/compare', form)
      setResult(res.data)
    } catch (e) {
      setError(e.response?.data?.error || e.userMessage || e.message || 'Comparison failed')
    } finally {
      setLoading(false)
    }
  }

  const imgUrl = (serverPath) => {
    if (!serverPath) return null
    const name = serverPath.split(/[/\\]/).pop()
    return `${BACKEND_URL}/api/scene-change/image/${name}`
  }

  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Scene Change Detection</h3>
      <p style={{ color: 'var(--muted)', fontSize: 14, marginBottom: 16 }}>
        Compare two views of the same location. The system aligns images, computes SSIM,
        highlights changed regions and reports a change score (possible object displacement).
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        <div>
          <label style={{ fontSize: 13, color: 'var(--muted)' }}>View A (reference)</label>
          <input type="file" accept="image/*" onChange={e => onPick('a', e)} style={{ display: 'block', marginTop: 6 }} />
          {previewA && <img src={previewA} alt="A" style={{ maxHeight: 160, marginTop: 8, borderRadius: 8, maxWidth: '100%' }} />}
        </div>
        <div>
          <label style={{ fontSize: 13, color: 'var(--muted)' }}>View B (compare)</label>
          <input type="file" accept="image/*" onChange={e => onPick('b', e)} style={{ display: 'block', marginTop: 6 }} />
          {previewB && <img src={previewB} alt="B" style={{ maxHeight: 160, marginTop: 8, borderRadius: 8, maxWidth: '100%' }} />}
        </div>
      </div>

      <button className="primary" onClick={compare} disabled={loading || !fileA || !fileB}>
        {loading ? 'Comparing scenes…' : 'Run Scene Change Detection'}
      </button>

      {error && <p style={{ color: 'var(--danger)', marginTop: 12, fontSize: 13 }}>{error}</p>}

      {result && (
        <div style={{ marginTop: 20 }}>
          <div className="grid grid-3" style={{ marginBottom: 16 }}>
            <div className="card" style={{ textAlign: 'center', padding: 12 }}>
              <div style={{ fontSize: 26, fontWeight: 700 }}>{result.change_score}</div>
              <div style={{ color: 'var(--muted)', fontSize: 12 }}>Change Score (0–100)</div>
            </div>
            <div className="card" style={{ textAlign: 'center', padding: 12 }}>
              <div style={{ fontSize: 26, fontWeight: 700 }}>{(result.ssim_score * 100).toFixed(1)}%</div>
              <div style={{ color: 'var(--muted)', fontSize: 12 }}>SSIM Similarity</div>
            </div>
            <div className="card" style={{ textAlign: 'center', padding: 12 }}>
              <div style={{ fontSize: 26, fontWeight: 700 }}>{result.region_count}</div>
              <div style={{ color: 'var(--muted)', fontSize: 12 }}>Changed Regions</div>
            </div>
          </div>

          <p style={{ fontSize: 14, marginBottom: 8 }}>
            <b>Interpretation:</b> {result.interpretation}
          </p>
          <p style={{ fontSize: 13, color: 'var(--muted)' }}>
            Alignment: {result.alignment_used ? `Yes (${result.alignment_inliers} inliers)` : 'Not applied / insufficient features'}
            {' · '}Changed pixels ratio: {(result.change_ratio * 100).toFixed(2)}%
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
            {result.overlay_image && (
              <div>
                <div style={{ fontSize: 13, marginBottom: 6, color: 'var(--muted)' }}>Changed regions overlay</div>
                <img src={imgUrl(result.overlay_image)} alt="overlay" style={{ width: '100%', borderRadius: 8 }} />
              </div>
            )}
            {result.heatmap_image && (
              <div>
                <div style={{ fontSize: 13, marginBottom: 6, color: 'var(--muted)' }}>Difference heatmap</div>
                <img src={imgUrl(result.heatmap_image)} alt="heatmap" style={{ width: '100%', borderRadius: 8 }} />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
