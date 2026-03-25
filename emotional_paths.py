"""
Goal-Based Emotional Regulation — Rule Engine Configuration.

Maps: emotion → paths (goal choices) → music_category / playlist.
UI-agnostic; used by the recommendation API to resolve playlists from emotion + path.
"""

from typing import TypedDict


class PathOption(TypedDict):
    """A single path (goal) the user can choose for an emotion."""
    id: str
    label: str
    description: str
    music_profile: str


class EmotionConfig(TypedDict):
    """Configuration for one emotion: prompt + list of path options."""
    prompt: str
    paths: list[PathOption]


# ---------------------------------------------------------------------------
# Rule engine: emotion → paths + prompt
# ---------------------------------------------------------------------------
EMOTION_PATHS: dict[str, EmotionConfig] = {
    "Angry": {
        "prompt": (
            "Feeling the heat? Do you want to blow off some steam with something "
            "heavy, or cool down and find some headspace?"
        ),
        "paths": [
            {
                "id": "lean_into_it",
                "label": "Lean Into It",
                "description": "Catharsis & Release",
                "music_profile": "aggressive beats, heavy bass, rock, metal, high tempo",
            },
            {
                "id": "cool_down",
                "label": "Cool Down",
                "description": "De-escalation",
                "music_profile": "lo-fi, ambient, slow jazz",
            },
        ],
    },
    "Neutral": {
        "prompt": (
            "Keeping it steady? Stay in the zone with focus-friendly tracks, "
            "or want a mood boost?"
        ),
        "paths": [
            {
                "id": "stay_steady",
                "label": "Stay Steady",
                "description": "Focus & Flow",
                "music_profile": "minimalist instrumentals, deep house, brown noise",
            },
            {
                "id": "shift_gears",
                "label": "Shift Gears",
                "description": "Inspiration",
                "music_profile": "indie, funk, discovery playlists",
            },
        ],
    },
    "Happy": {
        "prompt": (
            "You're glowing! Turn it up and keep the party going, "
            "or relax into a feel-good groove?"
        ),
        "paths": [
            {
                "id": "celebrate",
                "label": "Celebrate",
                "description": "Maximum Euphoria",
                "music_profile": "high-energy pop, dance anthems",
            },
            {
                "id": "mellow_out",
                "label": "Mellow Out",
                "description": "Contentment",
                "music_profile": "acoustic, chill lounge",
            },
        ],
    },
    "Fear": {
        "prompt": (
            "Feeling uneasy? Want music that understands you, "
            "or something grounding and calming?"
        ),
        "paths": [
            {
                "id": "validate",
                "label": "Validate",
                "description": "Emotional Solidarity",
                "music_profile": "melodic minor, atmospheric indie, vulnerable lyrics",
            },
            {
                "id": "ground",
                "label": "Ground",
                "description": "Security & Calm",
                "music_profile": "nature sounds, binaural beats, predictable classical",
            },
        ],
    },
    "Sad": {
        "prompt": (
            "Having a tough moment? Sit with it with reflective tunes, "
            "or lean into something warm and comforting?"
        ),
        "paths": [
            {
                "id": "lean_in",
                "label": "Lean In",
                "description": "Reflection",
                "music_profile": "melancholic, reflective, soft vocals",
            },
            {
                "id": "comfort",
                "label": "Comfort",
                "description": "Warmth & Ease",
                "music_profile": "warm acoustic, gentle piano, soothing",
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Playlist resolution: (emotion, path_id) → Spotify embed URL
# Add or replace with your own playlist IDs. Structure is easy to extend.
# ---------------------------------------------------------------------------
# Format: (emotion, path_id) -> Spotify embed playlist URL
SPOTIFY_BY_EMOTION_AND_PATH: dict[tuple[str, str], str] = {
    # Angry
    ("Angry", "lean_into_it"): "https://open.spotify.com/embed/playlist/37i9dQZF1DWXRqgorJj26U",   # Rock
    ("Angry", "cool_down"): "https://open.spotify.com/embed/playlist/37i9dQZF1DX4sWSpwq3LiO",     # Chill
    # Neutral
    ("Neutral", "stay_steady"): "https://open.spotify.com/embed/playlist/37i9dQZF1DX4sWSpwq3LiO", # Focus
    ("Neutral", "shift_gears"): "https://open.spotify.com/embed/playlist/37i9dQZF1DXdPec7aLTmlC", # Indie/Funk
    # Happy
    ("Happy", "celebrate"): "https://open.spotify.com/embed/playlist/37i9dQZF1DXdPec7aLTmlC",     # Pop/Dance
    ("Happy", "mellow_out"): "https://open.spotify.com/embed/playlist/37i9dQZF1DX4sWSpwq3LiO",    # Chill
    # Fear
    ("Fear", "validate"): "https://open.spotify.com/embed/playlist/37i9dQZF1DX7qK8ma5wgG1",       # Atmospheric
    ("Fear", "ground"): "https://open.spotify.com/embed/playlist/37i9dQZF1DWXe9gFZP0gtP",         # Calm
    # Sad
    ("Sad", "lean_in"): "https://open.spotify.com/embed/playlist/37i9dQZF1DX7qK8ma5wgG1",         # Sad
    ("Sad", "comfort"): "https://open.spotify.com/embed/playlist/37i9dQZF1DWXe9gFZP0gtP",         # Soothing
}

# Fallback when emotion has no path config or path_id is unknown
DEFAULT_PLAYLIST = "https://open.spotify.com/embed/playlist/37i9dQZF1DX4sWSpwq3LiO"


def get_paths_for_emotion(emotion: str) -> EmotionConfig | None:
    """
    Return path options and prompt for an emotion.
    Normalizes emotion string (strip, match key in EMOTION_PATHS).
    """
    if not emotion or not isinstance(emotion, str):
        return None
    key = emotion.strip()
    return EMOTION_PATHS.get(key)


def get_playlist_uri(emotion: str, path_id: str) -> str:
    """
    Resolve Spotify embed URI from emotion + path (goal).
    Returns DEFAULT_PLAYLIST if the pair is not in the rule table.
    """
    if not emotion or not path_id:
        return DEFAULT_PLAYLIST
    key = (emotion.strip(), path_id.strip())
    return SPOTIFY_BY_EMOTION_AND_PATH.get(key, DEFAULT_PLAYLIST)


def get_supported_emotions() -> list[str]:
    """Return list of emotions that have path configs (for validation/UI)."""
    return list(EMOTION_PATHS.keys())
