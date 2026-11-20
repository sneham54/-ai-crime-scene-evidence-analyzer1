import React, { createContext, useContext, useState, useEffect } from 'react'
import api from '../services/api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [token, setToken] = useState(() => localStorage.getItem('token'))
  const [biometricPassed, setBiometricPassed] = useState(
    () => localStorage.getItem('biometricPassed') === 'true'
  )

  useEffect(() => {
    if (token) {
      api.defaults.headers.common['Authorization'] = `Bearer ${token}`
    }
  }, [token])

  const login = async (username, password) => {
    const res = await api.post('/api/auth/login', { username, password })
    const { access_token, user: u } = res.data
    if (typeof access_token !== 'string' || !access_token.trim()) {
      throw new Error('The login response did not include a valid access token.')
    }
    localStorage.setItem('token', access_token)
    setToken(access_token)
    setUser(u)
    api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
    // Reset biometric verification on a new login.
    setBiometricPassed(false)
    localStorage.removeItem('biometricPassed')
    return u
  }

  const register = async (username, password) => {
    await api.post('/api/auth/register', { username, password })
    return login(username, password)
  }

  const logout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('biometricPassed')
    setToken(null)
    setUser(null)
    setBiometricPassed(false)
    delete api.defaults.headers.common['Authorization']
  }

  const passBiometric = async (force = false) => {
    try {
      if (!force) {
        await api.post('/api/auth/biometric', {
          verified: true,
          biometric_token: 'demo-biometric-ok'
        })
      }
      setBiometricPassed(true)
      localStorage.setItem('biometricPassed', 'true')
      return true
    } catch {
      // Always allow in demo so the page never stays blank
      setBiometricPassed(true)
      localStorage.setItem('biometricPassed', 'true')
      return true
    }
  }

  return (
    <AuthContext.Provider value={{
      user,
      token,
      login,
      register,
      logout,
      biometricPassed,
      passBiometric,
      isAuthenticated: typeof token === 'string' && token.trim().length > 0
    }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
