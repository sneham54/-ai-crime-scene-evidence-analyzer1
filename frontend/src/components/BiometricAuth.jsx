import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function BiometricAuth() {
  const { passBiometric, isAuthenticated, biometricPassed } = useAuth()
  const navigate = useNavigate()
  const [status, setStatus] = useState('Ready to scan')
  const [scanning, setScanning] = useState(false)

  // Redirect only inside effects (never during render)
  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login', { replace: true })
    }
  }, [isAuthenticated, navigate])

  useEffect(() => {
    if (biometricPassed) {
      navigate('/', { replace: true })
    }
  }, [biometricPassed, navigate])

  const handleScan = async () => {
    setScanning(true)
    setStatus('Scanning fingerprint...')
    await new Promise(r => setTimeout(r, 1200))

    try {
      const ok = await passBiometric()
      if (ok) {
        setStatus('Fingerprint verified successfully!')
        // navigation happens via the effect above
      } else {
        // Demo fallback – still allow entry
        setStatus('Verified (Demo mode)')
        // force pass
        await passBiometric(true)
      }
    } catch (err) {
      console.warn('Biometric API error – continuing in demo mode', err)
      setStatus('Verified (Demo mode)')
      await passBiometric(true)
    } finally {
      setScanning(false)
    }
  }

  if (!isAuthenticated) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#0a0e17', color: '#fff' }}>
        Redirecting to login...
      </div>
    )
  }

  return (
    <div style={{
      minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'radial-gradient(ellipse at bottom, #1a1a2e 0%, #0a0e17 70%)'
    }}>
      <div className="card" style={{ width: 400, textAlign: 'center' }}>
        <div style={{
          width: 100, height: 100, margin: '0 auto 20px', borderRadius: '50%',
          background: 'linear-gradient(135deg, #1e3a5f, #0d47a1)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 40
        }}>
          👆
        </div>
        <h2 style={{ margin: '0 0 8px' }}>Biometric Authentication</h2>
        <p style={{ color: 'var(--muted)', fontSize: 14, marginBottom: 24 }}>
          Place finger on the scanner to continue
        </p>

        <button className="primary" onClick={handleScan} disabled={scanning} style={{ width: '100%' }}>
          {scanning ? 'Scanning...' : 'Simulate Fingerprint Scan'}
        </button>

        {status && (
          <p style={{
            marginTop: 16,
            color: status.includes('verified') || status.includes('Demo') || status.includes('success')
              ? 'var(--success)' : 'var(--muted)'
          }}>
            {status}
          </p>
        )}
      </div>
    </div>
  )
}
