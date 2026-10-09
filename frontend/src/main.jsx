import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { AuthProvider } from './context/AuthContext'
import './index.css'

// Simple error boundary so a component crash never produces a pure blank page
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, message: '' }
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, message: error?.message || 'Unknown error' }
  }
  componentDidCatch(error, info) {
    console.error('UI Error:', error, info)
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          minHeight: '100vh', display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          background: '#0a0e17', color: '#e0e6f0', padding: 24, textAlign: 'center'
        }}>
          <h2 style={{ color: '#f87171' }}>Something went wrong</h2>
          <p style={{ color: '#94a3b8', maxWidth: 480 }}>{this.state.message}</p>
          <button
            onClick={() => {
              localStorage.clear()
              window.location.href = '/login'
            }}
            style={{
              marginTop: 16, padding: '10px 20px', background: '#3b82f6',
              color: 'white', border: 'none', borderRadius: 8, cursor: 'pointer'
            }}
          >
            Clear session & go to Login
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ErrorBoundary>
      <BrowserRouter>
        <AuthProvider>
          <App />
        </AuthProvider>
      </BrowserRouter>
    </ErrorBoundary>
  </React.StrictMode>
)
