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
            "What do you want next? Calm down or let it out?"
        ),
        "paths": [
            {
                "id": "lean_into_it",
                "label": "Let it out",
                "description": "Loud and strong",
                "music_profile": "aggressive beats, heavy bass, rock, metal, high tempo",
            },
            {
                "id": "cool_down",
                "label": "Calm down",
                "description": "Slow and steady",
                "music_profile": "lo-fi, ambient, slow jazz",
            },
        ],
    },
    "Neutral": {
        "prompt": (
            "Pick a goal: focus or feel better."
        ),
        "paths": [
            {
                "id": "",
                "label": "Focus",
                "description": "Calm and steady",
                "music_profile": "minimalist instrumentals, deep house, brown noise",
            },
            {
                "id": "shift_stay_steadygears",
                "label": "Boost",
                "description": "Light and fun",
                "music_profile": "indie, funk, discovery playlists",
            },
        ],
    },
    "Happy": {
        "prompt": (
            "Keep the good vibes, or slow it down."
        ),
        "paths": [
            {
                "id": "celebrate",
                "label": "Keep going",
                "description": "More energy",
                "music_profile": "high-energy pop, dance anthems",
            },
            {
                "id": "mellow_out",
                "label": "Chill",
                "description": "Relax",
                "music_profile": "acoustic, chill lounge",
            },
        ],
    },
    "Fear": {
        "prompt": (
            "Pick what feels safe: comfort or calm."
        ),
        "paths": [
            {
                "id": "validate",
                "label": "Be kind to me",
                "description": "Gentle support",
                "music_profile": "melodic minor, atmospheric indie, vulnerable lyrics",
            },
            {
                "id": "ground",
                "label": "Feel calm",
                "description": "Safe and steady",
                "music_profile": "nature sounds, binaural beats, predictable classical",
            },
        ],
    },
    "Sad": {
        "prompt": (
            "Pick: comfort, or a warm boost."
        ),
        "paths": [
            {
                "id": "lean_in",
                "label": "Feel it",
                "description": "Soft and sad",
                "music_profile": "melancholic, reflective, soft vocals",
            },
            {
                "id": "comfort",
                "label": "Be comforted",
                "description": "Warm and easy",
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
