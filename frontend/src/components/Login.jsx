import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function Login() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [mode, setMode] = useState('login')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [evidencePreview, setEvidencePreview] = useState(null)
  const { login, register } = useAuth()
  const navigate = useNavigate()
  const sessionExpired = new URLSearchParams(window.location.search).get('reason') === 'session-expired'

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      if (mode === 'register') {
        await register(username, password)
      } else {
        await login(username, password)
      }
      navigate('/')
    } catch (err) {
      setError(err.response?.data?.error || `${mode === 'register' ? 'Registration' : 'Login'} failed. Please try again.`)
    } finally {
      setLoading(false)
    }
  }

  const chooseEvidence = (event) => {
    const file = event.target.files?.[0]
    if (!file || !file.type.startsWith('image/')) return
    setEvidencePreview(URL.createObjectURL(file))
  }

  return (
    <div className="auth-page">
      <div className="auth-frame">
        <div className="auth-intro">
          <div className="auth-brand"><span className="brand-mark">+</span><span>FORENSIQ</span></div>
          <div className="eyebrow">AI crime scene evidence analyzer</div>
          <h1>Evidence intelligence for the field.</h1>
          <p>Object detection, video frame forensics, scene comparison, risk scoring, and case-ready reporting in one investigator console.</p>
          <div className="auth-features">
            {['YOLOv8 detection', 'Frame timeline', 'Scene change analysis', 'PDF reporting'].map(feature => <span key={feature}><i className="feature-check">+</i>{feature}</span>)}
          </div>
          <label className="login-evidence-drop">
            <input type="file" accept="image/*" onChange={chooseEvidence} />
            {evidencePreview ? <img src={evidencePreview} alt="Selected crime scene evidence" /> : <>
              <span className="drop-icon">+</span>
              <span><b>Have evidence to inspect?</b><small>Select a crime scene image to preview after sign in</small></span>
            </>}
          </label>
          <div className="auth-restricted"><i className="status-dot" /> Restricted system - all activity is logged</div>
        </div>
        <div className="card auth-card">
          <div className="section-kicker">Secure access</div>
          <h2>{mode === 'login' ? 'Investigator sign in' : 'Create investigator account'}</h2>
          <p className="muted-copy">Use your department credentials to continue.</p>

          {sessionExpired && <p role="status" style={{ color: 'var(--warning)', fontSize: 13 }}>
            Your session expired or its token is no longer valid. Please sign in again.
          </p>}

        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: 14 }}>
            <label style={{ fontSize: 13, color: 'var(--muted)' }}>Badge ID / Email</label>
            <input value={username} onChange={e => setUsername(e.target.value)} required />
          </div>
          <div style={{ marginBottom: 18 }}>
            <label style={{ fontSize: 13, color: 'var(--muted)' }}>Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} required />
          </div>

          {error && <p style={{ color: 'var(--danger)', fontSize: 13, marginBottom: 12 }}>{error}</p>}

          <button type="submit" className="primary" style={{ width: '100%' }} disabled={loading}>
            {loading ? (mode === 'login' ? 'Signing in...' : 'Creating account...') : (mode === 'login' ? 'Sign in' : 'Register')}
          </button>
        </form>

        {mode === 'login' ? (
          <>
            <button type="button" className="secondary" style={{ width: '100%' }} onClick={() => { setMode('register'); setError('') }}>
              Create an account
            </button>
          </>
        ) : (
          <button type="button" className="secondary" style={{ width: '100%', marginTop: 16 }} onClick={() => { setMode('login'); setError('') }}>
            Back to login
          </button>
        )}
        </div>
      </div>
    </div>
  )
}
