import axios from 'axios'

// Backend is usually at localhost:5000.
// If you run backend on another IP (e.g. 10.x.x.x), change this value.
const BACKEND_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000'
const DEFAULT_TIMEOUT = 20000
const LONG_TIMEOUT = 600000

const api = axios.create({
  baseURL: BACKEND_URL,
  timeout: DEFAULT_TIMEOUT,
})

export const longTimeoutApi = axios.create({
  baseURL: BACKEND_URL,
  timeout: LONG_TIMEOUT,
})

// Helpful debug
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const responseMessage = err.response?.data?.msg || err.response?.data?.error || ''
    if (err.response?.status === 422 && /token|jwt|signature/i.test(responseMessage)) {
      // Tokens from before a signing-key change cannot be used; clear them and
      // send the user back through login instead of leaving the app stuck at 422.
      localStorage.removeItem('token')
      localStorage.removeItem('biometricPassed')
      window.location.assign('/login')
    }
    if (err.code === 'ERR_NETWORK' || err.message?.includes('Network Error')) {
      err.userMessage = `Cannot reach backend at ${BACKEND_URL}. Is the Flask server running?`
    }
    if (err.response?.status === 503 && err.response?.data?.error) {
      err.userMessage = err.response.data.error
    }
    return Promise.reject(err)
  }
)

export default api
export { BACKEND_URL, DEFAULT_TIMEOUT, LONG_TIMEOUT }
