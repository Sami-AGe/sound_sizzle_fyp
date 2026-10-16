"""
AudioStory — LoRA Style + Mood Registry

Defines all available visual styles and mood-based colour grades.

STYLE_LORAS keys map to .safetensors files in ComfyUI/models/loras/ on the
Windows PC. Each entry provides LoRA metadata for local ComfyUI generation
plus Stage 2 prompt direction for the Groq storyboard LLM.

MOOD_GRADES maps detected moods to Stage 2 narrative direction and Stage 4
colour-grading parameters.

To add a new LoRA: drop .safetensors into ComfyUI/models/loras/, add one
entry to STYLE_LORAS, and the pipeline picks it up automatically.
"""

from typing import Any, Dict

# ─────────────────────────────────────────────────────────────────────────
# Visual Styles — each entry maps to a LoRA file for ComfyUI
# ─────────────────────────────────────────────────────────────────────────
STYLE_LORAS: Dict[str, Dict[str, Any]] = {
    "none": {
        "display_name": "Default",
        "description": "No style LoRA — standard anime illustration",
        "lora_file": None, "lora_weight": 0,    "trigger_word": "",
        "stage2_direction": "", "negative_add": ""
    },
    "mha": {
        "display_name": "My Hero Academia",
        "description": "Shonen anime style inspired by My Hero Academia",
        "lora_file": "AnimeKiss.safetensors",
        "lora_weight": 0.75,
        "trigger_word": "anime style, vibrant",
        "stage2_direction": "Scenes must be bold, dynamic, high-energy anime action.",
        "negative_add": "realistic, photographic",
    },
    "ghibli": {
        "display_name": "Studio Ghibli",
        "description": "Soft, painterly Studio Ghibli aesthetic",
        "lora_file": "bluestyle-000005.safetensors",
        "lora_weight": 0.7,
        "trigger_word": "ghibli style, painterly, soft",
        "stage2_direction": "Scenes must be whimsical, nature-focused, contemplative.",
        "negative_add": "dark, violent, harsh lighting",
    },
    "cyberpunk_anime": {
        "display_name": "Cyberpunk Anime",
        "description": "Neon-lit cyberpunk anime aesthetic",
        "lora_file": "Cyberpunk_Anime-10.safetensors",
        "lora_weight": 0.8,
        "trigger_word": "cyberpunk, neon lights, futuristic",
        "stage2_direction": "Scenes must be urban, neon-lit, rain-slicked streets.",
        "negative_add": "nature, daylight, pastoral",
    },
}

# ─────────────────────────────────────────────────────────────────────────
# Mood Grades — each entry drives Stage 2 direction + Stage 4 colour grade
# ─────────────────────────────────────────────────────────────────────────
MOOD_GRADES: Dict[str, Dict[str, Any]] = {
    "neutral": {
        "stage2_direction": "",
        "stage2_negative": "",
        "grade": None,
    },
    "neutral_cinematic": {
        "stage2_direction": "Cinematic pacing, wide shots, deliberate camera angles.",
        "stage2_negative": "chaotic, cluttered",
        "grade": None,
    },
    "energetic": {
        "stage2_direction": "Fast cuts, dynamic poses, high contrast lighting.",
        "stage2_negative": "slow, quiet, static",
        "grade": {"saturation": 1.3, "brightness": 1.1,
                  "color_shift": [1.05, 1.0, 0.95]},
    },
    "melancholic": {
        "stage2_direction": "Slow pacing, long holds, muted tones, rain or fog.",
        "stage2_negative": "bright, happy, energetic",
        "grade": {"saturation": 0.75, "brightness": 0.9,
                  "color_shift": [0.9, 0.95, 1.1]},
    },
    "euphoric": {
        "stage2_direction": "Warm light floods, wide smiles, upward camera angles.",
        "stage2_negative": "dark, somber, enclosed",
        "grade": {"saturation": 1.4, "brightness": 1.15,
                  "color_shift": [1.1, 1.05, 0.9]},
    },
    "tense": {
        "stage2_direction": "Tight close-ups, low angles, harsh shadows.",
        "stage2_negative": "open, peaceful, bright",
        "grade": {"saturation": 1.1, "brightness": 0.85,
                  "color_shift": [1.1, 0.95, 0.9]},
    },
    # Backward-compat: map old dark_intense to tense
    "dark_intense": {
        "stage2_direction": "Tight close-ups, low angles, harsh shadows.",
        "stage2_negative": "open, peaceful, bright",
        "grade": {"saturation": 1.1, "brightness": 0.85,
                  "color_shift": [1.1, 0.95, 0.9]},
    },
}

# Backward-compat aliases — old code imports LORA_STYLES
LORA_STYLES = STYLE_LORAS


def get_style(style_key: str) -> dict:
    """Returns the style configuration dictionary for a given style key."""
    if style_key not in STYLE_LORAS:
        raise ValueError(f"Unknown style: {style_key}. Available: {list(STYLE_LORAS.keys())}")
    return STYLE_LORAS[style_key]


def list_styles() -> dict:
    """
    Returns a simplified list of styles for the API / frontend.
    Each entry includes display_name, description, and lora_file.
    """
    return {
        key: {
            "display_name": s["display_name"],
            "description": s["description"],
            "lora_file": s["lora_file"],
            "lora_weight": s["lora_weight"],
        }
        for key, s in STYLE_LORAS.items()
    }


def get_recommended_style(mood: str) -> str:
    """
    Returns the best style key for a given audio mood.
    Falls back to 'none' as a safe default for cinematic/neutral moods.
    """
    mood_style_map = {
        "melancholic": "ghibli",
        "energetic": "mha",
        "euphoric": "mha",
        "dark_intense": "cyberpunk_anime",
        "tense": "cyberpunk_anime",
        "neutral": "none",
        "neutral_cinematic": "none",
    }
    return mood_style_map.get(mood, "none")
