export function normalizeVideoEvents(result = {}) {
  const classification = result.classification || result.event_classification || {}
  const events = result.activity_events || classification.activity_events || result.events || classification.events || []
  return events.map(event => ({
    ...event,
    category: event.category || event.predicted_class,
    start_timestamp_sec: event.start_timestamp_sec ?? event.start_timestamp,
    end_timestamp_sec: event.end_timestamp_sec ?? event.end_timestamp,
    verification_status: event.verification_status || 'pending_review',
  }))
}
