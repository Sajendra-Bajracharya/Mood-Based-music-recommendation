import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import Webcam from 'react-webcam'
import axios from 'axios'

const API_BASE_URL = 'http://localhost:5000'

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
  const [paths, setPaths] = useState([])
  const [prompt, setPrompt] = useState(null)
  const [spotifyUri, setSpotifyUri] = useState(null)
  const [pathLoading, setPathLoading] = useState(false)
  const [toast, setToast] = useState({ show: false, message: '' })

  const webcamRef = useRef(null)
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

  const handleCapture = async () => {
    if (!webcamRef.current || typeof webcamRef.current.getScreenshot !== 'function') {
      setToast({ show: true, message: 'Webcam not ready. Please allow camera access and try again.' })
      return
    }

    const imageSrc = webcamRef.current.getScreenshot()
    if (!imageSrc) {
      setToast({ show: true, message: 'No frame captured. Please try again.' })
      return
    }

    setStatus(STATUS.ANALYZING)
    setSpotifyUri(null)

    try {
      const response = await axios.post(
        `${API_BASE_URL}/api/analyze`,
        { image: imageSrc },
        { headers: { 'Content-Type': 'application/json' } },
      )

      const {
        emotion: respEmotion,
        cluster: respCluster,
        paths: respPaths,
        prompt: respPrompt,
        spotify_uri: respSpotifyUri,
      } = response.data

      if (!respEmotion) {
        setStatus(STATUS.ERROR)
        setToast({ show: true, message: 'No face detected or model could not determine emotion. Please try again.' })
        return
      }

      setEmotion(respEmotion)
      setCluster(respCluster ?? null)
      if (respSpotifyUri) {
        setSpotifyUri(respSpotifyUri)
        setPaths([])
        setPrompt(null)
      } else {
        setPaths(Array.isArray(respPaths) ? respPaths : [])
        setPrompt(respPrompt ?? null)
      }
      setStatus(STATUS.SUCCESS)
    } catch (error) {
      const backendMessage = error.response?.data?.error
      setStatus(STATUS.ERROR)
      setToast({
        show: true,
        message: backendMessage || 'Failed to analyze image. Ensure the backend is running and try again.',
      })
    }
  }

  const handleSelectPath = async (pathId) => {
    if (!emotion || !pathId) return
    setPathLoading(true)
    try {
      const response = await axios.post(
        `${API_BASE_URL}/api/recommend`,
        { emotion, path_id: pathId },
        { headers: { 'Content-Type': 'application/json' } },
      )
      const uri = response.data?.spotify_uri
      if (uri) setSpotifyUri(uri)
      else setToast({ show: true, message: 'Could not load playlist.' })
    } catch (error) {
      setToast({
        show: true,
        message: error.response?.data?.error || 'Failed to load recommendation.',
      })
    } finally {
      setPathLoading(false)
    }
  }

  const statusColor =
    status === STATUS.SUCCESS ? 'text-emerald-400' : status === STATUS.ERROR ? 'text-rose-400' : 'text-amber-300'

  const hasPaths = emotion && paths?.length > 0

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
            <p className="mt-1 text-sm text-slate-300">Cluster: <span className="text-white">{cluster ?? '--'}</span></p>
            <button
              type="button"
              onClick={handleCapture}
              disabled={status === STATUS.ANALYZING}
              className="mt-4 w-full rounded-xl bg-indigo-500 px-4 py-3 font-semibold text-white transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {status === STATUS.ANALYZING ? 'Analyzing...' : 'Capture'}
            </button>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
          <h3 className="text-lg font-semibold">Music Recommendation</h3>

          {!hasPaths && !spotifyUri && (
            <p className="mt-3 text-sm text-slate-300">Capture your mood to get a playlist recommendation.</p>
          )}

          {hasPaths && (
            <div className="mt-3 space-y-2">
              <p className="text-xs text-slate-300">Detected: {emotion}</p>
              <p className="text-sm text-white">{prompt}</p>
              {paths.map((path) => (
                <button
                  key={path.id}
                  type="button"
                  onClick={() => handleSelectPath(path.id)}
                  disabled={pathLoading}
                  className="w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-left transition hover:border-indigo-500 disabled:cursor-not-allowed disabled:opacity-70"
                >
                  <p className="font-semibold text-white">{path.label}</p>
                  <p className="text-sm text-slate-300">{path.description}</p>
                </button>
              ))}
              {pathLoading && <p className="text-sm text-amber-300">Loading playlist...</p>}
            </div>
          )}

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