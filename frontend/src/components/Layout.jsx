import React, { useEffect, useState } from 'react'
import { useAuth } from '../context/AuthContext'

export default function Layout({ children }) {
  const { user, logout } = useAuth()
  const [activeTab, setActiveTab] = useState('overview')
  const goTo = tab => {
    setActiveTab(tab)
    window.dispatchEvent(new CustomEvent('forensiq:navigate', { detail: tab }))
  }

  useEffect(() => {
    const handleTabChange = event => setActiveTab(event.detail)
    window.addEventListener('forensiq:tab-changed', handleTabChange)
    return () => window.removeEventListener('forensiq:tab-changed', handleTabChange)
  }, [])

  const navItems = [
    ['overview', '◫', 'Dashboard'],
    ['upload', '▧', 'Image Analysis'],
    ['video', '▶', 'Video Analysis'],
    ['results', '⌗', 'Detected Objects'],
    ['scene', '◉', 'Scene Change'],
    ['risk', '△', 'Risk Analysis'],
    ['history', '▤', 'Case History'],
    ['reports', '▣', 'Reports'],
    ['chatbot', '◇', 'AI Assistant'],
  ]

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <span className="brand-mark">+</span>
          <div>
            <div className="brand-kicker">Forensics AI</div>
            <div className="brand-name">Evidence Analyzer</div>
          </div>
        </div>
        <div className="sidebar-rule" />
        <div className="sidebar-label">Investigation console</div>
        <div className="sidebar-list">
          {navItems.map(([tab, icon, label]) => <button key={tab} className={`sidebar-item ${activeTab === tab || (tab === 'upload' && activeTab === 'results') ? 'active' : ''}`} onClick={() => goTo(tab)}><span className="sidebar-icon">{icon}</span>{label}</button>)}
        </div>
        <div className="sidebar-footer">
          <div className="security-note"><i className="status-dot" /> System online<br /><span>YOLOv8 inference ready</span></div>
          <button className="secondary sidebar-logout" onClick={logout}>Sign out</button>
        </div>
      </aside>
      <div className="workspace-shell">
        <header className="topbar">
          <div className="mobile-brand"><span className="brand-mark">+</span> FORENSIQ</div>
          <div className="topbar-meta"><span className="topbar-status"><i className="status-dot" /> Analysis system online</span><span>{user?.username || 'Investigator'}</span></div>
        </header>
        <nav className="mobile-nav" aria-label="Main navigation">
          {navItems.map(([tab, icon, label]) => <button key={tab} className={activeTab === tab ? 'active' : ''} onClick={() => goTo(tab)}><span>{icon}</span>{label}</button>)}
        </nav>
        <main className="workspace-main">{children}</main>
      </div>
    </div>
  )
}
