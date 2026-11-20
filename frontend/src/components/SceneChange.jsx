import React, { useEffect, useState } from 'react'
import api, { BACKEND_URL } from '../services/api'

/**
 * Compare two images from the same CCTV scene.
 * The server runs the existing SSIM/alignment comparison and, when its
 * checkpoint is available, an experimental independent BEFORE/AFTER classifier.
 */
export default function SceneChange() {
  const [fileA, setFileA] = useState(null)
  const [fileB, setFileB] = useState(null)
  const [previewA, setPreviewA] = useState(null)
  const [previewB, setPreviewB] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    return () => {
      if (previewA) URL.revokeObjectURL(previewA)
      if (previewB) URL.revokeObjectURL(previewB)
    }
  }, [previewA, previewB])

  const onPick = (which, event) => {
    const file = event.target.files?.[0]
    if (!file) return

    if (!file.type.startsWith('image/')) {
      setError('Please select valid image files.')
      event.target.value = ''
      return
    }

    const url = URL.createObjectURL(file)

    if (which === 'a') {
      setFileA(file)
      setPreviewA(url)
    } else {
      setFileB(file)
      setPreviewB(url)
    }

    setResult(null)
    setError('')
  }

  const compare = async () => {
    if (!fileA || !fileB) {
      setError('Please select both a Before image and an After image.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const form = new FormData()
      form.append('file_a', fileA)
      form.append('file_b', fileB)

      const response = await api.post('/api/scene-change/compare', form)
      setResult(response.data)
    } catch (requestError) {
      setError(
        requestError.response?.data?.error ||
        requestError.userMessage ||
        requestError.message ||
        'Scene comparison failed.'
      )
    } finally {
      setLoading(false)
    }
  }

  const imageUrl = (serverPath) => {
    if (!serverPath) return null
    const name = serverPath.split(/[/\\]/).pop()
    return `${BACKEND_URL}/api/scene-change/image/${encodeURIComponent(name)}`
  }

  const classifier = result?.screenshot_classifier

  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Before / After Scene Comparison</h3>
      <p style={{ color: 'var(--muted)', fontSize: 14, marginBottom: 16 }}>
        Select two images of the same CCTV location. The existing comparison
        aligns the images when possible, calculates SSIM and pixel differences,
        and highlights candidate changed regions.
      </p>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
        gap: 16,
        marginBottom: 16
      }}>
        <div>
          <label style={{ fontSize: 13, color: 'var(--muted)' }}>
            Before image (reference)
          </label>
          <input
            type="file"
            accept="image/png,image/jpeg,image/jpg,image/webp,image/bmp"
            onChange={event => onPick('a', event)}
            style={{ display: 'block', marginTop: 6, maxWidth: '100%' }}
          />
          {previewA && (
            <img
              src={previewA}
              alt="Before scene"
              style={{ maxHeight: 220, marginTop: 8, borderRadius: 8, maxWidth: '100%' }}
            />
          )}
          {fileA && <div style={{ fontSize: 12, color: 'var(--muted)' }}>{fileA.name}</div>}
        </div>

        <div>
          <label style={{ fontSize: 13, color: 'var(--muted)' }}>
            After image (comparison)
          </label>
          <input
            type="file"
            accept="image/png,image/jpeg,image/jpg,image/webp,image/bmp"
            onChange={event => onPick('b', event)}
            style={{ display: 'block', marginTop: 6, maxWidth: '100%' }}
          />
          {previewB && (
            <img
              src={previewB}
              alt="After scene"
              style={{ maxHeight: 220, marginTop: 8, borderRadius: 8, maxWidth: '100%' }}
            />
          )}
          {fileB && <div style={{ fontSize: 12, color: 'var(--muted)' }}>{fileB.name}</div>}
        </div>
      </div>

      <button className="primary" onClick={compare} disabled={loading || !fileA || !fileB}>
        {loading ? 'Comparing images…' : 'Compare Before and After'}
      </button>

      {error && (
        <p style={{ color: 'var(--danger)', marginTop: 12, fontSize: 13 }}>{error}</p>
      )}

      {result && (
        <div style={{ marginTop: 20 }}>
          <h4>Image comparison results</h4>

          <div className="grid grid-3" style={{ marginBottom: 16 }}>
            <div className="card" style={{ textAlign: 'center', padding: 12 }}>
              <div style={{ fontSize: 26, fontWeight: 700 }}>{result.change_score}</div>
              <div style={{ color: 'var(--muted)', fontSize: 12 }}>Change score (0–100)</div>
            </div>
            <div className="card" style={{ textAlign: 'center', padding: 12 }}>
              <div style={{ fontSize: 26, fontWeight: 700 }}>
                {Number.isFinite(result.ssim_score) ? `${(result.ssim_score * 100).toFixed(1)}%` : 'N/A'}
              </div>
              <div style={{ color: 'var(--muted)', fontSize: 12 }}>SSIM similarity</div>
            </div>
            <div className="card" style={{ textAlign: 'center', padding: 12 }}>
              <div style={{ fontSize: 26, fontWeight: 700 }}>{result.region_count}</div>
              <div style={{ color: 'var(--muted)', fontSize: 12 }}>Candidate changed regions</div>
            </div>
          </div>

          <p style={{ fontSize: 14, marginBottom: 8 }}>
            <b>Image-difference interpretation:</b> {result.interpretation}
          </p>
          <p style={{ fontSize: 13, color: 'var(--muted)' }}>
            Alignment: {result.alignment_used
              ? `Yes (${result.alignment_inliers} inliers)`
              : 'Not applied / insufficient matching features'}
            {' · '}Changed pixels: {(Number(result.change_ratio || 0) * 100).toFixed(2)}%
          </p>

          {classifier && (
            <div className="card" style={{ marginTop: 18, padding: 14 }}>
              <h4 style={{ marginTop: 0 }}>Experimental trained screenshot classifier</h4>

              {classifier.status === 'available' ? (
                <>
                  <p style={{ fontSize: 13 }}>
                    <b>Before upload prediction:</b>{' '}
                    {classifier.reference_image?.predicted_class || 'Unavailable'}
                    {classifier.reference_image?.softmax_score != null
                      ? ` (score ${(classifier.reference_image.softmax_score * 100).toFixed(1)}%)`
                      : ''}
                  </p>
                  <p style={{ fontSize: 13 }}>
                    <b>After upload prediction:</b>{' '}
                    {classifier.comparison_image?.predicted_class || 'Unavailable'}
                    {classifier.comparison_image?.softmax_score != null
                      ? ` (score ${(classifier.comparison_image.softmax_score * 100).toFixed(1)}%)`
                      : ''}
                  </p>
                  <p style={{ fontSize: 12, color: 'var(--muted)' }}>
                    This ResNet18 was trained to classify each screenshot independently
                    as BEFORE or AFTER. It does not identify the type of scene change.
                    Softmax scores are not calibrated accuracy or forensic evidence.
                  </p>
                </>
              ) : (
                <p style={{ fontSize: 13, color: 'var(--muted)' }}>
                  Screenshot classifier status: {classifier.status}.
                  {classifier.message ? ` ${classifier.message}` : ''}
                  The image-difference comparison above is still available.
                </p>
              )}
            </div>
          )}

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: 16,
            marginTop: 16
          }}>
            {result.overlay_image && (
              <div>
                <div style={{ fontSize: 13, marginBottom: 6, color: 'var(--muted)' }}>
                  Candidate changed regions
                </div>
                <img
                  src={imageUrl(result.overlay_image)}
                  alt="Candidate changed regions overlay"
                  style={{ width: '100%', borderRadius: 8 }}
                />
              </div>
            )}
            {result.heatmap_image && (
              <div>
                <div style={{ fontSize: 13, marginBottom: 6, color: 'var(--muted)' }}>
                  Difference heatmap
                </div>
                <img
                  src={imageUrl(result.heatmap_image)}
                  alt="Image difference heatmap"
                  style={{ width: '100%', borderRadius: 8 }}
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
