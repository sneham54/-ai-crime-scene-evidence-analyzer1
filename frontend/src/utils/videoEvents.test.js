import test from 'node:test'
import assert from 'node:assert/strict'
import { normalizeVideoEvents } from './videoEvents.js'

test('normalizes experimental event schema for the existing video timeline', () => {
  const keyframes = [{ frame_index: 45, timestamp_sec: 1.5 }]
  const [event] = normalizeVideoEvents({ events: [{
    predicted_class: 'Assault', start_timestamp: 1, end_timestamp: 1.9,
    confidence: 0.85, keyframes,
  }] })
  assert.equal(event.category, 'Assault')
  assert.equal(event.start_timestamp_sec, 1)
  assert.equal(event.end_timestamp_sec, 1.9)
  assert.equal(event.verification_status, 'pending_review')
  assert.equal(event.keyframes, keyframes)
})

test('keeps current production activity event fields intact', () => {
  const [event] = normalizeVideoEvents({ activity_events: [{
    category: 'Robbery', start_timestamp_sec: 3, end_timestamp_sec: 7,
    verification_status: 'unverified_candidate',
  }] })
  assert.equal(event.category, 'Robbery')
  assert.equal(event.start_timestamp_sec, 3)
  assert.equal(event.end_timestamp_sec, 7)
  assert.equal(event.verification_status, 'unverified_candidate')
})
