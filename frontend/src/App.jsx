import { useEffect, useMemo, useRef, useState } from 'react'
import Webcam from 'react-webcam'
import axios from 'axios'
import 'bootstrap/dist/css/bootstrap.min.css'
import './App.css'

const API_BASE_URL = 'http://localhost:5000'

const STATUS = {
  IDLE: 'Idle',
  ANALYZING: 'Analyzing',
  SUCCESS: 'Success',
  ERROR: 'Error',
}

function Header() {
  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark border-bottom border-secondary mb-4">
      <div className="container-fluid">
        <span className="navbar-brand fw-bold">Melo</span>
      </div>
    </nav>
  )
}

function CameraView({ webcamRef }) {
  const videoConstraints = useMemo(
    () => ({
      width: 480,
      height: 360,
      facingMode: 'user',
    }),
    [],
  )

  return (
    <div className="camera-wrapper bg-dark rounded-3 p-2 border border-secondary">
      <Webcam
        ref={webcamRef}
        audio={false}
        screenshotFormat="image/jpeg"
        videoConstraints={videoConstraints}
        className="w-100 rounded-3"
      />
    </div>
  )
}

function AnalysisCard({ status, emotion, cluster, onCapture, disabled }) {
  const isAnalyzing = status === STATUS.ANALYZING

  return (
    <div className="card bg-dark text-light border-secondary shadow-sm h-100">
      <div className="card-header border-secondary">
        <h5 className="mb-0">Mood Analysis</h5>
      </div>
      <div className="card-body">
        <p className="text-muted mb-2">
          Status:{' '}
          <span
            className={
              status === STATUS.SUCCESS
                ? 'text-success'
                : status === STATUS.ERROR
                  ? 'text-danger'
                  : 'text-warning'
            }
          >
            {status}
          </span>
        </p>

        <p className="mb-1">
          <span className="fw-semibold">Detected Emotion:</span>{' '}
          {emotion ?? '--'}
        </p>
        <p className="mb-3">
          <span className="fw-semibold">Cluster:</span>{' '}
          {cluster ?? '--'}
        </p>

        <button
          type="button"
          className="btn btn-primary w-100"
          onClick={onCapture}
          disabled={disabled || isAnalyzing}
        >
          {isAnalyzing ? 'Analyzing...' : 'Capture'}
        </button>
      </div>
    </div>
  )
}

function MusicPlayer({ spotifyUri }) {
  if (!spotifyUri) {
    return (
      <div className="card bg-dark text-light border-secondary shadow-sm h-100">
        <div className="card-header border-secondary">
          <h5 className="mb-0">Music Recommendation</h5>
        </div>
        <div className="card-body d-flex align-items-center justify-content-center">
          <p className="text-muted mb-0">
            Capture your mood to get a playlist recommendation.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="card bg-dark text-light border-secondary shadow-sm h-100 music-card">
      <div className="card-header border-secondary">
        <h5 className="mb-0">Music Recommendation</h5>
      </div>
      <div className="card-body p-0">
        <div className="music-embed-wrapper">
          <iframe
            src={spotifyUri}
            title="Spotify Player"
            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
            loading="lazy"
            className="music-embed-iframe"
          />
        </div>
      </div>
    </div>
  )
}

function ToastAlert({ show, message, onClose }) {
  if (!show) return null

  return (
    <div
      className="toast-container position-fixed bottom-0 end-0 p-3"
      style={{ zIndex: 1080 }}
    >
      <div
        className="toast show text-bg-danger border-0"
        role="alert"
        aria-live="assertive"
        aria-atomic="true"
      >
        <div className="toast-header">
          <strong className="me-auto">Melo</strong>
          <button
            type="button"
            className="btn-close btn-close-white"
            aria-label="Close"
            onClick={onClose}
          />
        </div>
        <div className="toast-body">{message}</div>
      </div>
    </div>
  )
}

function App() {
  const [status, setStatus] = useState(STATUS.IDLE)
  const [emotion, setEmotion] = useState(null)
  const [cluster, setCluster] = useState(null)
  const [spotifyUri, setSpotifyUri] = useState(null)
  const [toast, setToast] = useState({ show: false, message: '' })

  const webcamRef = useRef(null)

  useEffect(() => {
    document.title = `Melo - ${status}`
  }, [status])

  // Auto-hide toast after 3 seconds whenever it appears
  useEffect(() => {
    if (!toast.show) return
    const timer = setTimeout(() => {
      setToast((prev) => ({ ...prev, show: false }))
    }, 3000)
    return () => clearTimeout(timer)
  }, [toast.show])

  const handleCapture = async () => {
    if (!webcamRef.current || typeof webcamRef.current.getScreenshot !== 'function') {
      setToast({
        show: true,
        message: 'Webcam not ready. Please allow camera access and try again.',
      })
      return
    }

    const imageSrc = webcamRef.current.getScreenshot()

    if (!imageSrc) {
      setToast({
        show: true,
        message: 'No frame captured. Please try again.',
      })
      return
    }

    setStatus(STATUS.ANALYZING)

    try {
      const response = await axios.post(
        `${API_BASE_URL}/api/recommend`,
        { image: imageSrc },
        {
          headers: {
            'Content-Type': 'application/json',
          },
        },
      )

      const { emotion: respEmotion, cluster: respCluster, spotify_uri } = response.data

      if (!respEmotion || !spotify_uri) {
        setStatus(STATUS.ERROR)
        setToast({
          show: true,
          message:
            'No face detected or model could not determine emotion. Please try again.',
        })
        return
      }

      setEmotion(respEmotion)
      setCluster(respCluster)
      setSpotifyUri(spotify_uri)
      setStatus(STATUS.SUCCESS)
    } catch (error) {
      console.error(error)
      const backendMessage = error.response?.data?.error
      setStatus(STATUS.ERROR)
      setToast({
        show: true,
        message:
          backendMessage ||
          'Failed to analyze image. Ensure the backend is running and try again.',
      })
    }
  }

  const closeToast = () => {
    setToast((prev) => ({ ...prev, show: false }))
  }

  return (
    <div className="bg-black min-vh-100 text-light">
      <Header />

      <div className="container py-4">
        <div className="row g-4">
          <div className="col-lg-6 d-flex flex-column gap-3">
            <CameraView webcamRef={webcamRef} />
            <AnalysisCard
              status={status}
              emotion={emotion}
              cluster={cluster}
              onCapture={handleCapture}
              disabled={false}
            />
          </div>

          <div className="col-lg-6">
            <MusicPlayer spotifyUri={spotifyUri} />
          </div>
        </div>
      </div>

      <ToastAlert
        show={toast.show}
        message={toast.message}
        onClose={closeToast}
      />
    </div>
  )
}

export default App
