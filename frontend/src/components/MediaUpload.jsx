import React, { useState, useRef } from 'react'
import api, { longTimeoutApi } from '../services/api'

async function extractFrameThumbnails(videoUrl, frames, fps, duration) {
  if (!videoUrl || !frames?.length) return []
  const video = document.createElement('video')
  video.src = videoUrl
  video.muted = true
  video.playsInline = true
  video.preload = 'auto'
  try {
    await new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve
      video.onerror = reject
    })
    const canvas = document.createElement('canvas')
    const context = canvas.getContext('2d')
    const scale = Math.min(1, 320 / (video.videoWidth || 320))
    canvas.width = Math.max(1, Math.round((video.videoWidth || 320) * scale))
    canvas.height = Math.max(1, Math.round((video.videoHeight || 180) * scale))
    const thumbnails = []
    for (const frame of frames) {
      const frameTime = frame.timestamp_sec ?? (fps ? frame.frame_number / fps : 0)
      const target = Math.min(Math.max(0, frameTime), Math.max(0, duration || video.duration || 0))
      await new Promise(resolve => {
        if (Math.abs(video.currentTime - target) < 0.01) return resolve()
        video.onseeked = resolve
        video.currentTime = target
      })
      context.drawImage(video, 0, 0, canvas.width, canvas.height)
      thumbnails.push({ frame_number: frame.frame_number, image: canvas.toDataURL('image/jpeg', 0.72) })
    }
    return thumbnails
  } catch {
    return []
  } finally {
    video.removeAttribute('src')
    video.load()
  }
}

export default function MediaUpload({ onComplete }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [statusMsg, setStatusMsg] = useState('')
  const inputRef = useRef(null)

  const onFileChange = (e) => {
    const selectedFile = e.target.files?.[0]

    if (!selectedFile) return

    setFile(selectedFile)
    setError('')
    setStatusMsg('')

    setPreview(URL.createObjectURL(selectedFile))
  }

  const analyze = async () => {
    if (!file) {
      setError('Please select an image or video first.')
      return
    }

    setLoading(true)
    setError('')
    setStatusMsg('Uploading...')

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

      const form = new FormData()
      form.append('file', file)

      const extension = file.name.split('.').pop()?.toLowerCase()
      const isVideo = file.type.startsWith('video/') || ['mp4', 'avi', 'mov', 'mkv', 'webm'].includes(extension)

      const uploadEndpoint = isVideo
        ? '/api/upload/video'
        : '/api/upload/image'

      const uploadResponse = await api.post(
        uploadEndpoint,
        form,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      )

      const filepath = uploadResponse.data.filepath
      const evidenceId = uploadResponse.data.evidence_id

      if (!filepath) {
        throw new Error('Backend did not return an uploaded file path.')
      }

      /*
       * STEP 2:
       * Analyze the uploaded media.
       */

      if (isVideo) {
        setStatusMsg('Processing video...')

        const attemptVideoAnalysis = async (framesToUse) => {
          return longTimeoutApi.post(
            '/api/video/analyze',
            {
              filepath,
              evidence_id: evidenceId,
              max_frames: framesToUse
            }
          )
        }

        let videoResponse
        try {
          videoResponse = await attemptVideoAnalysis(12)
        } catch (videoErr) {
          const timeoutLike =
            videoErr.code === 'ECONNABORTED' ||
            videoErr.message?.toLowerCase().includes('timeout') ||
            videoErr.message?.toLowerCase().includes('timed out')

          if (timeoutLike) {
            setStatusMsg('Video is taking longer on CPU; retrying with minimum frames...')
            videoResponse = await attemptVideoAnalysis(6)
          } else {
            throw videoErr
          }
        }

        const frameThumbnails = await extractFrameThumbnails(
          preview,
          videoResponse.data.frame_results || [],
          videoResponse.data.video_info?.fps,
          videoResponse.data.video_info?.duration_sec
        )
        setStatusMsg('Video analysis complete!')
        onComplete?.({
          ...videoResponse.data,
          file_name: uploadResponse.data.original_name || file.name,
          media_preview: preview,
          frame_thumbnails: frameThumbnails,
          analyzed_at: new Date().toISOString(),
          evidence_id: evidenceId,
        })

      } else {
        setStatusMsg('Running YOLO detection...')

        const detectionResponse = await api.post(
          '/api/detection/analyze',
          {
            filepath,
            evidence_id: evidenceId
          }
        )

        setStatusMsg('Image analysis complete!')
        onComplete?.({
          ...detectionResponse.data,
          file_name: uploadResponse.data.original_name || file.name,
          analyzed_at: new Date().toISOString(),
          evidence_id: evidenceId,
        })
      }

    } catch (err) {
      console.error('ANALYSIS ERROR:', err)

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

    } finally {
      setLoading(false)
    }
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
        Supported: JPG, PNG, BMP, WEBP, MP4, AVI.
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
          accept="image/*,video/*"
          onChange={onFileChange}
          style={{ display: 'none' }}
        />

        {preview && file?.type.startsWith('video/') ? (
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
