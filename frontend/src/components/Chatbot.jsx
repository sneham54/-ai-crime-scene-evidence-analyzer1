import React, { useState, useRef, useEffect } from 'react'
import { longTimeoutApi } from '../services/api'

export default function Chatbot({ context }) {
  const [messages, setMessages] = useState([
    { role: 'bot', text: 'Hello Investigator. I can summarize the available case context, explain AI findings, organize evidence, and suggest follow-up questions. I will distinguish reported observations from interpretation and identify when case data is insufficient.' }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [health, setHealth] = useState({ status: 'checking', error: 'Checking Ollama connection…' })
  const bottomRef = useRef(null)

  const checkHealth = async () => {
    setHealth({ status: 'checking', error: 'Checking Ollama connection…' })
    try {
      const res = await longTimeoutApi.get('/api/chatbot/health', { timeout: 7000 })
      setHealth(res.data)
    } catch (e) {
      setHealth(e.response?.data || { status: 'unavailable', error: e.userMessage || 'Could not check Ollama. Check the backend connection.' })
    }
  }

  useEffect(() => { checkHealth() }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async () => {
    if (!input.trim()) return
    const userMsg = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', text: userMsg }])
    setLoading(true)

    try {
      const res = await longTimeoutApi.post('/api/chatbot/query', {
        query: userMsg,
        context
      })
      setMessages(prev => [...prev, { role: 'bot', text: res.data.answer }])
    } catch (e) {
      const timedOut = e.code === 'ECONNABORTED' || e.message?.toLowerCase().includes('timeout')
      const message = e.response?.data?.error || e.userMessage || (timedOut
        ? 'The assistant is still generating a response. Please retry in a moment.'
        : 'The AI assistant could not process that request. Please check that the Flask backend and Ollama are running.')
      setMessages(prev => [...prev, { role: 'bot', text: message }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', height: 480 }}>
      <h3 style={{ marginTop: 0 }}>AI Forensic Chatbot</h3>
      <div role="status" style={{ fontSize: 12, color: health.status === 'connected' ? 'var(--success, #4ade80)' : 'var(--muted)', marginBottom: 8 }}>
        {health.status === 'connected' ? `Ollama connected · ${health.model}` : health.error || 'Ollama status unavailable'}
        {health.status !== 'connected' && <button type="button" className="text-button" onClick={checkHealth} style={{ marginLeft: 8 }}>Check again</button>}
      </div>

      <div style={{
        flex: 1, overflowY: 'auto', background: '#0f172a', borderRadius: 8,
        padding: 12, marginBottom: 12
      }}>
        {messages.map((m, i) => (
          <div key={i} style={{
            marginBottom: 10,
            textAlign: m.role === 'user' ? 'right' : 'left'
          }}>
            <span style={{
              display: 'inline-block',
              maxWidth: '80%',
              padding: '8px 12px',
              borderRadius: 12,
              background: m.role === 'user' ? 'var(--primary)' : '#1e293b',
              fontSize: 13,
              whiteSpace: 'pre-wrap'
            }}>
              {m.text}
            </span>
          </div>
        ))}
        {loading && <div style={{ color: 'var(--muted)', fontSize: 13 }}>Thinking...</div>}
        <div ref={bottomRef} />
      </div>

      <div style={{ display: 'flex', gap: 8 }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send()}
          placeholder="Ask about this case, its evidence, or next investigative steps..."
        />
        <button className="primary" onClick={send} disabled={loading}>Send</button>
      </div>
    </div>
  )
}
