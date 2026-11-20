
import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './context/AuthContext'
import Login from './components/Login'
import BiometricAuth from './components/BiometricAuth'
import Dashboard from './components/Dashboard'
import Layout from './components/Layout'
import AssaultDetections from './components/AssaultDetections'

function PrivateRoute({ children }) {
  const { isAuthenticated } = useAuth()

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  return children
}

export default function App() {
  return (
    <Routes>
      {/* Login page */}
      <Route
        path="/login"
        element={<Login />}
      />

      {/* Biometric authentication page */}
      <Route
        path="/biometric"
        element={<BiometricAuth />}
      />

      {/* Main dashboard */}
      <Route
        path="/"
        element={
          <PrivateRoute>
            <Layout>
              <Dashboard />
            </Layout>
          </PrivateRoute>
        }
      />

      {/* Assault video detection timestamps */}
      <Route
        path="/assault-detections"
        element={
          <PrivateRoute>
            <Layout>
              <AssaultDetections />
            </Layout>
          </PrivateRoute>
        }
      />

      {/* Redirect unknown routes */}
      <Route
        path="*"
        element={<Navigate to="/" replace />}
      />
    </Routes>
  )
}
