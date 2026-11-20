import axios from 'axios'

// Empty base URL keeps browser calls same-origin; Vite proxies /api to Flask.
const BACKEND_URL = import.meta.env.VITE_API_URL || ''
const DEFAULT_TIMEOUT = 20000
const IMAGE_ANALYSIS_TIMEOUT = 60000
const LONG_TIMEOUT = 600000

const api = axios.create({
  baseURL: BACKEND_URL,
  timeout: DEFAULT_TIMEOUT,
})

function attachStoredBearerToken(client) {
  client.interceptors.request.use((config) => {
    const token = localStorage.getItem('token')?.trim()
    if (token) {
      config.headers = config.headers || {}
      config.headers.Authorization = `Bearer ${token}`
    } else if (config.headers?.Authorization) {
      delete config.headers.Authorization
    }
    return config
  })
}

export const longTimeoutApi = axios.create({
  baseURL: BACKEND_URL,
  timeout: LONG_TIMEOUT,
})

export const imageAnalysisApi = axios.create({
  baseURL: BACKEND_URL,
  timeout: IMAGE_ANALYSIS_TIMEOUT,
})

attachStoredBearerToken(api)
attachStoredBearerToken(longTimeoutApi)
attachStoredBearerToken(imageAnalysisApi)

function handleApiError(err) {
  const responseMessage = err.response?.data?.error || err.response?.data?.msg || ''
  const authCode = err.response?.data?.code
  const authFailure = ['authentication_required', 'token_expired', 'invalid_token'].includes(authCode) ||
    (err.response?.status === 422 && /token|jwt|signature/i.test(responseMessage))
  if (authFailure) {
    localStorage.removeItem('token')
    localStorage.removeItem('biometricPassed')
    err.userMessage = responseMessage || 'Your session has expired. Please sign in again.'
    if (!window.location.pathname.startsWith('/login')) {
      window.location.assign('/login?reason=session-expired')
    }
  }
  if (err.code === 'ERR_NETWORK' || err.message?.includes('Network Error')) {
    err.userMessage = BACKEND_URL
      ? `Cannot reach Flask API at ${BACKEND_URL}. Check the Flask server and network settings.`
      : 'Cannot reach the Flask API through the /api proxy. Check that Flask and the Vite proxy are running.'
  }
  if (err.response?.status === 503 && err.response?.data?.error) {
    err.userMessage = err.response.data.error
  }
  return Promise.reject(err)
}

api.interceptors.response.use((res) => res, handleApiError)
longTimeoutApi.interceptors.response.use((res) => res, handleApiError)
imageAnalysisApi.interceptors.response.use((res) => res, handleApiError)

export default api
export { BACKEND_URL, DEFAULT_TIMEOUT, IMAGE_ANALYSIS_TIMEOUT, LONG_TIMEOUT }
