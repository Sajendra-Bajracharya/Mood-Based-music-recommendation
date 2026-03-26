import { Navigate, Route, Routes } from 'react-router-dom'
import LandingPage from './LandingPage'
import MusicRecommender from './MusicRecommender'

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/app" element={<MusicRecommender />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App
