import { useNavigate } from 'react-router-dom'
import { useState, useEffect } from 'react'

const carouselImages = [
  {
    url: 'https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=600&q=80',
    label: 'Live Performance',
  },
  {
    url: 'https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=600&q=80',
    label: 'Feel the Beat',
  },
  {
    url: 'https://images.unsplash.com/photo-1459749411175-04bf5292ceea?w=600&q=80',
    label: 'Concert Vibes',
  },
  {
    url: 'https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=600&q=80',
    label: 'DJ Set',
  },
]

function LandingPage() {
  const navigate = useNavigate()
  const [current, setCurrent] = useState(0)

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrent((prev) => (prev + 1) % carouselImages.length)
    }, 3000)
    return () => clearInterval(timer)
  }, [])

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50">
      <nav className="sticky top-0 z-40 border-b border-slate-800 bg-slate-950/90 backdrop-blur">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-4">
          <div className="text-xl font-bold tracking-wide">Melo</div>
          <div className="flex items-center gap-6 text-sm text-slate-300">
            <a href="#features" className="transition hover:text-slate-50">Features</a>
            <a href="#about" className="transition hover:text-slate-50">About</a>
          </div>
        </div>
      </nav>

      <main>
        <section className="mx-auto flex min-h-[72vh] w-full max-w-6xl flex-col items-center justify-center gap-12 px-6 py-16 md:flex-row">
          {/* Left: Text Content */}
          <div className="flex flex-1 flex-col justify-center">
            <h1 className="max-w-4xl text-4xl font-extrabold leading-tight md:text-6xl">
              Unlock Your Sound Through Your Soul.
            </h1>
            <p className="mt-6 max-w-2xl text-base text-slate-300 md:text-lg">
              Melo captures your facial expression through your camera, identifies your current mood, and suggests music that matches how you feel. Instead of searching for songs manually, it instantly recommends playlists tailored to your emotions.
            </p>
            <div className="mt-10">
              <button
                type="button"
                onClick={() => navigate('/app')}
                className="rounded-xl bg-violet-500 px-8 py-4 text-lg font-semibold text-white shadow-[0_0_30px_rgba(139,92,246,0.55)] transition hover:bg-violet-400"
              >
                Try Now
              </button>
            </div>
          </div>

          {/* Right: Carousel */}
          <div className="relative flex w-full flex-1 flex-col items-center">
            <div className="relative h-80 w-full overflow-hidden rounded-2xl border border-slate-800 bg-black shadow-[0_0_40px_rgba(0,0,0,0.8)] md:h-96">
              {carouselImages.map((img, i) => (
                <div
                  key={i}
                  className="absolute inset-0 transition-opacity duration-700"
                  style={{ opacity: i === current ? 1 : 0 }}
                >
                  <img
                    src={img.url}
                    alt={img.label}
                    className="h-full w-full object-cover opacity-80"
                  />
                  {/* Dark overlay */}
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
                  {/* Label */}
                  <p className="absolute bottom-4 left-5 text-sm font-medium tracking-widest text-slate-300 uppercase">
                    {img.label}
                  </p>
                </div>
              ))}
            </div>

            {/* Dot indicators */}
            <div className="mt-4 flex gap-2">
              {carouselImages.map((_, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => setCurrent(i)}
                  className={`h-2 rounded-full transition-all duration-300 ${
                    i === current
                      ? 'w-6 bg-violet-500'
                      : 'w-2 bg-slate-600 hover:bg-slate-400'
                  }`}
                />
              ))}
            </div>
          </div>
        </section>

        <section id="features" className="mx-auto w-full max-w-6xl px-6 py-10">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
              <h3 className="mb-3 text-lg font-semibold text-slate-50">Real-time Face Capture</h3>
              <p className="text-sm leading-relaxed text-slate-400">
                Melo instantly accesses your device camera and analyzes your facial expressions in real time. No uploads or manual input needed — just look at the screen and let the AI do the work.
              </p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
              <h3 className="mb-3 text-lg font-semibold text-slate-50">Emotion-aware Song Matching</h3>
              <p className="text-sm leading-relaxed text-slate-400">
                Our AI maps your detected emotion — whether happy, calm, sad, or energized — to a curated set of tracks that perfectly complement your current state of mind.
              </p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
              <h3 className="mb-3 text-lg font-semibold text-slate-50">Spotify Playlist Integration</h3>
              <p className="text-sm leading-relaxed text-slate-400">
                Melo connects directly with Spotify to pull personalized playlists based on your mood. Your music lives where it always has — we just make the discovery smarter.
              </p>
            </div>
          </div>
        </section>

        <section id="about" className="mx-auto w-full max-w-6xl px-6 py-14 text-slate-300">
          Melo blends deep learning and music personalization to quickly match songs with your present mood.
        </section>
      </main>

      <footer className="border-t border-slate-800 px-6 py-8">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-4 text-sm text-slate-400 md:flex-row md:items-center md:justify-between">
          <p>Made for music lovers</p>
          <div className="flex items-center gap-4">
            <a href="#" className="transition hover:text-slate-200">X</a>
            <a href="#" className="transition hover:text-slate-200">Instagram</a>
            <a href="#" className="transition hover:text-slate-200">YouTube</a>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default LandingPage