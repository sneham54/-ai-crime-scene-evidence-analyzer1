import test from 'node:test'
import assert from 'node:assert/strict'
import { extractFrameThumbnails, waitForMediaEvent } from './videoThumbnails.js'

class FakeVideo {
  constructor(seekEvents = true) {
    this.seekEvents = seekEvents
    this.listeners = new Map()
    this.currentTime = 0
    this.videoWidth = 640
    this.videoHeight = 480
    this.duration = 30
    this.removedSource = false
  }

  addEventListener(name, callback) {
    const listeners = this.listeners.get(name) || new Set()
    listeners.add(callback)
    this.listeners.set(name, listeners)
  }

  removeEventListener(name, callback) {
    this.listeners.get(name)?.delete(callback)
  }

  dispatch(name) {
    for (const listener of this.listeners.get(name) || []) listener()
  }

  load() {
    if (this.src) queueMicrotask(() => this.dispatch('loadedmetadata'))
  }

  set currentTime(value) {
    this._currentTime = value
    if (value !== 0 && this.seekEvents) queueMicrotask(() => this.dispatch('seeked'))
  }

  get currentTime() {
    return this._currentTime || 0
  }

  removeAttribute(name) {
    if (name === 'src') {
      this.src = ''
      this.removedSource = true
    }
  }
}

function fakeDocument({ seekEvents = true } = {}) {
  let video
  return {
    get video() { return video },
    createElement(name) {
      if (name === 'video') {
        video = new FakeVideo(seekEvents)
        return video
      }
      return {
        width: 0,
        height: 0,
        getContext: () => ({ drawImage() {} }),
        toDataURL: () => 'data:image/jpeg;base64,test',
      }
    },
  }
}

test('thumbnail extraction is capped to the six frames displayed in analysis', async () => {
  const documentRef = fakeDocument()
  const frames = Array.from({ length: 30 }, (_, frame_number) => ({
    frame_number,
    timestamp_sec: frame_number,
  }))
  const thumbnails = await extractFrameThumbnails('blob:video', frames, 1, 30, documentRef)
  assert.equal(thumbnails.length, 6)
  assert.equal(documentRef.video.removedSource, true)
})

test('missing media events reject after the configured timeout and clear listeners', async () => {
  const media = new FakeVideo()
  await assert.rejects(waitForMediaEvent(media, 'seeked', 5), /Timed out/)
  assert.equal(media.listeners.get('seeked').size, 0)
  assert.equal(media.listeners.get('error').size, 0)
})

test('thumbnail extraction obeys one total deadline and releases the video source', async () => {
  const documentRef = fakeDocument({ seekEvents: false })
  const started = Date.now()
  const thumbnails = await extractFrameThumbnails(
    'blob:video', [{ frame_number: 1, timestamp_sec: 1 }, { frame_number: 2, timestamp_sec: 2 }],
    1, 30, documentRef, 40
  )
  assert.deepEqual(thumbnails, [])
  assert.ok(Date.now() - started < 500)
  assert.equal(documentRef.video.removedSource, true)
})
