import React, { useState, useRef, useEffect } from 'react'
import { imageAnalysisApi, longTimeoutApi } from '../services/api'
import { useAuth } from '../context/AuthContext'
import { extractFrameThumbnails } from '../utils/videoThumbnails'
import { withLoadingState } from '../utils/loadingState'

function isVideoFile(file) {
  const extension = file?.name?.split('.').pop()?.toLowerCase()
  return file?.type?.startsWith('video/') || ['mp4', 'avi', 'mov', 'mkv', 'webm'].includes(extension)
}

export default function MediaUpload({ onComplete }) {
  const { isAuthenticated } = useAuth()
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [statusMsg, setStatusMsg] = useState('')
  const [analyzeEveryFrame, setAnalyzeEveryFrame] = useState(true)
  const [videoCategory, setVideoCategory] = useState('')
  const [caseId, setCaseId] = useState('')
  const inputRef = useRef(null)
  const previewUrlRef = useRef(null)

  useEffect(() => () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current)
  }, [])

  const onFileChange = (e) => {
    const selectedFile = e.target.files?.[0]

    if (!selectedFile) return

    setFile(selectedFile)
    setVideoCategory('')
    setError('')
    setStatusMsg('')

    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current)
    previewUrlRef.current = URL.createObjectURL(selectedFile)
    setPreview(previewUrlRef.current)
  }

  const analyze = async () => {
    if (!isAuthenticated || !localStorage.getItem('token')?.trim()) {
      setError('Please sign in again before analyzing evidence.')
      return
    }
    if (!file) {
      setError('Please select an image or video first.')
      return
    }
    if (isVideoFile(file) && !videoCategory) {
      setError('Select the source video category. Normal videos are classified too; saved tracking lookup applies only where inventory exists.')
      return
    }

    setError('')
    setStatusMsg(isVideoFile(file) ? 'Uploading...' : 'Uploading and analyzing image...')

    await withLoadingState(setLoading, async () => {
      try {
      /*
       * STEP 1:
       * Upload the file using the correct backend endpoint.
       *
       * Image:
       *   /api/upload/image
       *
       * Video:
       *   /api/upload/video
       */

      const isVideo = isVideoFile(file)
      if (isVideo) {
        const form = new FormData()
        form.append('file', file)
        if (caseId.trim()) form.append('case_id', caseId.trim())
        const uploadResponse = await longTimeoutApi.post('/api/upload/video', form)
        const filepath = uploadResponse.data.filepath
        const evidenceId = uploadResponse.data.evidence_id
        if (!filepath) throw new Error('Backend did not return an uploaded file path.')

        setStatusMsg(analyzeEveryFrame
          ? 'Analyzing every frame up to the configured duration limit...'
          : 'Analyzing sampled frames and matching saved tracking evidence...')

        const videoResponse = await longTimeoutApi.post(
          '/api/video/analyze',
          { filepath, evidence_id: evidenceId, full_video: analyzeEveryFrame,
            category: videoCategory, case_id: caseId.trim() || undefined,
            original_filename: uploadResponse.data.original_name || file.name }
        )

        const frameThumbnails = await extractFrameThumbnails(
          preview,
          videoResponse.data.frame_results || [],
          videoResponse.data.video_info?.fps,
          videoResponse.data.video_info?.duration_sec
        )
        setStatusMsg('Video analysis complete!')
        // Dashboard owns this object URL after the upload panel unmounts.
        // Keep it alive so the original, unannotated upload remains playable.
        const mediaPreview = preview
        previewUrlRef.current = null
        onComplete?.({
          ...videoResponse.data,
          file_name: uploadResponse.data.original_name || file.name,
          uploaded_video_filepath: filepath,
          media_preview: mediaPreview,
          frame_thumbnails: frameThumbnails,
          analyzed_at: new Date().toISOString(),
          evidence_id: evidenceId,
        })

      } else {
        const evidenceId = window.crypto?.randomUUID?.() || `image-${Date.now()}-${Math.random().toString(16).slice(2)}`
        const imageForm = new FormData()
        imageForm.append('file', file)
        imageForm.append('evidence_id', evidenceId)
        imageForm.append('compare', 'true')
        if (caseId.trim()) imageForm.append('case_id', caseId.trim())

        setStatusMsg('Uploading image and running YOLO analysis...')
        const requestStarted = performance.now()
        console.info('[image-analysis] request started')
        const detectionResponse = await imageAnalysisApi.post('/api/detection/analyze', imageForm, {
          onUploadProgress: event => {
            if (event.total && event.loaded >= event.total) {
              console.info('[image-analysis] file upload sent to Flask')
            }
          },
        })
        console.info('[image-analysis] request completed', {
          durationMs: Math.round(performance.now() - requestStarted),
          detectionCount: detectionResponse.data.detection_count,
        })

        setStatusMsg('Image analysis complete!')
        onComplete?.({
          ...detectionResponse.data,
          file_name: file.name,
          analyzed_at: new Date().toISOString(),
          evidence_id: evidenceId,
        })
      }

      } catch (err) {
        console.error('[media-analysis] request failed', {
          status: err.response?.status,
          code: err.code,
          message: err.response?.data?.error || err.message,
        })
        setStatusMsg('')

        const backendMessage =
          err.response?.data?.error ||
          err.response?.data?.message ||
          err.response?.data?.hint

        if (backendMessage) {
          setError(backendMessage)
        } else if (err.userMessage) {
          setError(err.userMessage)
        } else if (err.message) {
          setError(err.message)
        } else {
          setError(
            'Analysis failed. Check the backend terminal for the actual error.'
          )
        }

      }
    })
  }

  return (
    <div>
      <h3 style={{ marginTop: 0 }}>
        Upload Crime Scene Media
      </h3>

      <p
        style={{
          color: 'var(--muted)',
          fontSize: 14
        }}
      >
        Supported: JPG, PNG, BMP, WEBP, MP4, AVI, MOV.
        <br />
        Images use your trained YOLO model.
        Videos are processed by the video analysis service.
      </p>

      <div
        onClick={() => inputRef.current?.click()}
        style={{
          border: '2px dashed var(--border)',
          borderRadius: 12,
          padding: 32,
          textAlign: 'center',
          cursor: 'pointer',
          marginBottom: 16,
          background: '#0f172a'
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*,video/*,.mp4,.avi,.mov,.mkv,.webm"
          onChange={onFileChange}
          style={{ display: 'none' }}
        />

        {preview && isVideoFile(file) ? (
          <video
            src={preview}
            controls
            onClick={event => event.stopPropagation()}
            style={{ maxHeight: 360, maxWidth: '100%', borderRadius: 10 }}
          />
        ) : preview ? (
          <img
            src={preview}
            alt="Crime scene preview"
            style={{
              maxHeight: 220,
              maxWidth: '100%',
              borderRadius: 8
            }}
          />
        ) : (
          <div>
            <div
              style={{
                fontSize: 40,
                marginBottom: 8
              }}
            >
              📁
            </div>

            <div>
              Click to select image or video
            </div>

            {file && (
              <div
                style={{
                  marginTop: 8,
                  color: 'var(--primary)'
                }}
              >
                {file.name}
              </div>
            )}
          </div>
        )}
      </div>

      <label className="form-group" style={{ display: 'block', marginBottom: 16 }}>
        <span>Investigation case ID (optional)</span>
        <input value={caseId} onChange={event => setCaseId(event.target.value)} placeholder="Paste a saved case ID to link this evidence" />
      </label>

      {file && (
        <p
          style={{
            fontSize: 13,
            color: 'var(--muted)'
          }}
        >
          Selected: {file.name}
          {' '}
          ({(file.size / 1024).toFixed(1)} KB)
        </p>
      )}

      <button
        className="primary"
        onClick={analyze}
        disabled={loading || !file}
      >
        {loading
          ? statusMsg || 'Analyzing...'
          : 'Analyze media'}
      </button>

      {file && isVideoFile(file) && (
        <label className="video-category-option">
          Source video category
          <select value={videoCategory} onChange={event => setVideoCategory(event.target.value)} disabled={loading}>
            <option value="">Select category</option>
            <option value="assault">Assault</option>
            <option value="robbery">Robbery</option>
            <option value="shooting">Shooting</option>
            <option value="normal">Normal</option>
          </select>
        </label>
      )}

      {file && isVideoFile(file) && (
        <label className="video-full-analysis-option">
          <input
            type="checkbox"
            checked={analyzeEveryFrame}
            onChange={event => setAnalyzeEveryFrame(event.target.checked)}
            disabled={loading}
          />
          Analyze every frame (slower; full-video analysis is used by default)
        </label>
      )}

      {error && (
        <p
          style={{
            color: 'var(--danger)',
            marginTop: 12,
            fontSize: 14
          }}
        >
          {error}
        </p>
      )}

      {statusMsg && !error && !loading && (
        <p
          style={{
            color: 'var(--success)',
            marginTop: 12,
            fontSize: 14
          }}
        >
          {statusMsg}
        </p>
      )}
    </div>
  )
}
