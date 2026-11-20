const MAX_THUMBNAILS = 6
const METADATA_TIMEOUT_MS = 10000
const SEEK_TIMEOUT_MS = 3000
const TOTAL_TIMEOUT_MS = 5000

function waitForMediaEvent(media, successEvent, timeoutMs) {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timer)
      media.removeEventListener(successEvent, onSuccess)
      media.removeEventListener('error', onError)
    }
    const onSuccess = () => {
      cleanup()
      resolve()
    }
    const onError = () => {
      cleanup()
      reject(new Error(`Video ${successEvent} failed`))
    }
    const timer = setTimeout(() => {
      cleanup()
      reject(new Error(`Timed out waiting for video ${successEvent}`))
    }, timeoutMs)
    media.addEventListener(successEvent, onSuccess, { once: true })
    media.addEventListener('error', onError, { once: true })
  })
}

function selectFrames(frames) {
  if (frames.length <= MAX_THUMBNAILS) return frames
  return Array.from({ length: MAX_THUMBNAILS }, (_, index) =>
    frames[Math.round(index * (frames.length - 1) / (MAX_THUMBNAILS - 1))]
  )
}

export async function extractFrameThumbnails(videoUrl, frames, fps, duration, documentRef = globalThis.document, timeoutMs = TOTAL_TIMEOUT_MS) {
  if (!videoUrl || !frames?.length || !documentRef) return []
  if (!(fps > 0) && frames.some(frame => frame.timestamp_sec == null)) return []

  const video = documentRef.createElement('video')
  video.src = videoUrl
  video.muted = true
  video.playsInline = true
  video.preload = 'auto'
  const deadline = Date.now() + Math.max(1, timeoutMs)
  const remainingTime = () => Math.max(0, deadline - Date.now())

  try {
    const metadataReady = waitForMediaEvent(video, 'loadedmetadata', Math.min(METADATA_TIMEOUT_MS, remainingTime()))
    video.load()
    await metadataReady

    const canvas = documentRef.createElement('canvas')
    const context = canvas.getContext('2d')
    if (!context) return []
    const scale = Math.min(1, 320 / (video.videoWidth || 320))
    canvas.width = Math.max(1, Math.round((video.videoWidth || 320) * scale))
    canvas.height = Math.max(1, Math.round((video.videoHeight || 180) * scale))
    const thumbnails = []

    for (const frame of selectFrames(frames)) {
      if (remainingTime() <= 0) break
      const frameTime = frame.timestamp_sec ?? (fps ? frame.frame_number / fps : 0)
      const target = Math.min(Math.max(0, frameTime), Math.max(0, duration || video.duration || 0))
      if (Math.abs(video.currentTime - target) >= 0.01) {
        try {
          const seekComplete = waitForMediaEvent(video, 'seeked', Math.min(SEEK_TIMEOUT_MS, remainingTime()))
          video.currentTime = target
          await seekComplete
        } catch {
          // A missing seek event must not hold up or fail the completed analysis.
          continue
        }
      }
      try {
        context.drawImage(video, 0, 0, canvas.width, canvas.height)
        thumbnails.push({ frame_number: frame.frame_number, image: canvas.toDataURL('image/jpeg', 0.72) })
      } catch {
        // Browser thumbnail generation is optional; keep any thumbnails already made.
      }
    }
    return thumbnails
  } catch {
    return []
  } finally {
    video.removeAttribute('src')
    video.load()
  }
}

export { waitForMediaEvent }
