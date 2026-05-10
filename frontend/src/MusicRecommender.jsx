import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import Webcam from 'react-webcam'
import axios from 'axios'

const API_BASE_URL = 'http://localhost:5000'
const CAPTURE_INTERVAL_MS = 5000
const MOOD_HISTORY_SIZE = 3

const STATUS = {
  IDLE: 'Idle',
  ANALYZING: 'Analyzing',
  SUCCESS: 'Success',
  ERROR: 'Error',
}

function MusicRecommender() {
  const [status, setStatus] = useState(STATUS.IDLE)
  const [emotion, setEmotion] = useState(null)
  const [cluster, setCluster] = useState(null)
  const [spotifyUri, setSpotifyUri] = useState(null)
  const [pathLoading, setPathLoading] = useState(false)
  const [toast, setToast] = useState({ show: false, message: '' })
  const [isMonitoring, setIsMonitoring] = useState(false)
  const [stableMood, setStableMood] = useState(null)

  const webcamRef = useRef(null)
  const intervalRef = useRef(null)
  const isAnalyzingRef = useRef(false)
  const moodHistoryRef = useRef([])
  const lastRecommendedMoodRef = useRef(null)
  const lastToastAtRef = useRef(0)
  const videoConstraints = useMemo(
    () => ({ width: 480, height: 360, facingMode: 'user' }),
    [],
  )

  useEffect(() => {
    document.title = `Melo - ${status}`
  }, [status])

  useEffect(() => {
    if (!toast.show) return undefined
    const timer = setTimeout(() => setToast((prev) => ({ ...prev, show: false })), 3000)
    return () => clearTimeout(timer)
  }, [toast.show])

  const pushToast = useCallback((message) => {
    const now = Date.now()
    if (now - lastToastAtRef.current < 3000) return
    lastToastAtRef.current = now
    setToast({ show: true, message })
  }, [])

  const pickStableMood = useCallback((history) => {
    if (history.length === 0) return null

    const counts = history.reduce((acc, mood) => {
      acc[mood] = (acc[mood] || 0) + 1
      return acc
    }, {})

    // Tie-break by favoring the most recent mood in history.
    let bestMood = history[history.length - 1]
    let bestCount = counts[bestMood]
    Object.entries(counts).forEach(([mood, count]) => {
      if (count > bestCount) {
        bestMood = mood
        bestCount = count
      }
    })
    return bestMood
  }, [])

  const fetchRecommendation = useCallback(async (targetMood, directSpotifyUri) => {
    if (directSpotifyUri) {
      setSpotifyUri(directSpotifyUri)
      return
    }

    setSpotifyUri(null)
    pushToast(`No recommendation mapping for ${targetMood}.`)
  }, [pushToast])

  const analyzeCurrentFrame = useCallback(async (showCameraErrorToast = true) => {
    if (isAnalyzingRef.current) return

    if (!webcamRef.current || typeof webcamRef.current.getScreenshot !== 'function') {
      if (showCameraErrorToast) {
        pushToast('Webcam not ready. Please allow camera access and try again.')
      }
      return
    }

    const imageSrc = webcamRef.current.getScreenshot()
    if (!imageSrc) {
      if (showCameraErrorToast) {
        pushToast('No frame captured. Please try again.')
      }
      return
    }

    isAnalyzingRef.current = true
    setStatus(STATUS.ANALYZING)

    try {
      const response = await axios.post(
        `${API_BASE_URL}/api/analyze`,
        { image: imageSrc },
        { headers: { 'Content-Type': 'application/json' } },
      )

      const {
        emotion: respEmotion,
        cluster: respCluster,
        spotify_uri: respSpotifyUri,
      } = response.data

      if (!respEmotion) {
        setStatus(STATUS.ERROR)
        if (showCameraErrorToast) {
          pushToast('No face detected or model could not determine emotion. Please try again.')
        }
        return
      }

      setEmotion(respEmotion)
      setCluster(respCluster ?? null)

      moodHistoryRef.current = [...moodHistoryRef.current, respEmotion].slice(-MOOD_HISTORY_SIZE)
      const nextStableMood = pickStableMood(moodHistoryRef.current)
      setStableMood(nextStableMood)

      if (nextStableMood && nextStableMood !== lastRecommendedMoodRef.current) {
        lastRecommendedMoodRef.current = nextStableMood
        await fetchRecommendation(nextStableMood, respSpotifyUri)
      }

      setStatus(STATUS.SUCCESS)
    } catch (error) {
      const backendMessage = error.response?.data?.error
      setStatus(STATUS.ERROR)
      if (showCameraErrorToast) {
        pushToast(backendMessage || 'Failed to analyze image. Ensure the backend is running and try again.')
      }
    } finally {
      isAnalyzingRef.current = false
    }
  }, [fetchRecommendation, pickStableMood, pushToast])

  useEffect(() => {
    if (!isMonitoring) return undefined

    analyzeCurrentFrame(false)
    intervalRef.current = setInterval(() => {
      analyzeCurrentFrame(false)
    }, CAPTURE_INTERVAL_MS)

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
  }, [analyzeCurrentFrame, isMonitoring])

  useEffect(() => () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }
  }, [])

  const statusColor =
    status === STATUS.SUCCESS ? 'text-emerald-400' : status === STATUS.ERROR ? 'text-rose-400' : 'text-amber-300'

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50">
      <nav className="fixed top-0 z-40 w-full border-b border-slate-800 bg-slate-950/90 backdrop-blur">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-4">
          <span className="text-xl font-bold">Melo</span>
          <Link
            to="/"
            className="rounded-lg border border-slate-700 px-4 py-2 text-sm font-medium text-slate-200 transition hover:border-slate-500 hover:text-white"
          >
            Back to Home
          </Link>
        </div>
      </nav>

      <div className="mx-auto grid w-full max-w-6xl gap-6 px-6 py-6 pt-20 md:grid-cols-2">
        <div className="space-y-4">
          <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900 p-2">
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              videoConstraints={videoConstraints}
              className="w-full rounded-xl"
            />
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
            <h3 className="text-lg font-semibold">Mood Analysis</h3>
            <p className="mt-2 text-sm text-slate-300">
              Status: <span className={statusColor}>{status}</span>
            </p>
            <p className="mt-1 text-sm text-slate-300">Detected Emotion: <span className="text-white">{emotion ?? '--'}</span></p>
            <p className="mt-1 text-sm text-slate-300">Stable Mood: <span className="text-white">{stableMood ?? '--'}</span></p>
            <p className="mt-1 text-sm text-slate-300">Cluster: <span className="text-white">{cluster ?? '--'}</span></p>
            <div className="mt-4 grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setIsMonitoring((prev) => !prev)}
                className="w-full rounded-xl bg-violet-500 px-4 py-3 text-sm font-semibold text-white transition hover:bg-violet-400"
              >
                {isMonitoring ? 'Stop Auto' : 'Start Auto'}
              </button>
              <button
                type="button"
                onClick={() => analyzeCurrentFrame(true)}
                disabled={status === STATUS.ANALYZING}
                className="w-full rounded-xl bg-indigo-500 px-4 py-3 text-sm font-semibold text-white transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-70"
              >
                {status === STATUS.ANALYZING ? 'Analyzing...' : 'Capture Now'}
              </button>
            </div>
            <p className="mt-2 text-xs text-slate-400">
              Auto mode captures every {CAPTURE_INTERVAL_MS / 1000} seconds and updates only when mood changes.
            </p>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
          <h3 className="text-lg font-semibold">Music Recommendation</h3>

          {!spotifyUri && !pathLoading && (
            <p className="mt-3 text-sm text-slate-300">Capture your mood to get a playlist recommendation.</p>
          )}

          {pathLoading && <p className="mt-3 text-sm text-amber-300">Updating recommendation...</p>}

          {spotifyUri && (
            <div className="mt-4 overflow-hidden rounded-xl border border-slate-800">
              <iframe
                src={spotifyUri}
                title="Spotify Player"
                allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
                loading="lazy"
                className="h-[420px] w-full border-0"
              />
            </div>
          )}
        </div>
      </div>

      {toast.show && (
        <div className="fixed bottom-5 right-5 rounded-lg border border-rose-500/50 bg-rose-600/90 px-4 py-3 text-sm text-white shadow-xl">
          {toast.message}
        </div>
      )}
    </div>
  )
}

export default MusicRecommender