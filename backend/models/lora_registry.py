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
        "display_name": "No Style (Base Model)",
        "description": "Base model without LoRA enhancement",
        "lora_file": None,
        "lora_weight": 0.0,
        "trigger_word": "anime style",
        "stage2_direction": "",
        "negative_add": "",
    },
    "mha": {
        "display_name": "My Hero Academia",
        "description": "Shonen anime style inspired by My Hero Academia",
        "lora_file": "mha_style_v2.safetensors",
        "lora_weight": 0.75,
        "trigger_word": "bnha style, boku no hero academia",
        "stage2_direction": "anime superhero aesthetic, bold outlines, dynamic action poses, vivid energy effects",
        "negative_add": "",
    },
    "ghibli": {
        "display_name": "Studio Ghibli",
        "description": "Soft, painterly Studio Ghibli aesthetic",
        "lora_file": "ghibli_style_v1.safetensors",
        "lora_weight": 0.70,
        "trigger_word": "ghibli style, studio ghibli, soft lighting",
        "stage2_direction": "painterly backgrounds, soft lighting, nature-focused, warm and whimsical atmosphere",
        "negative_add": "dark, gritty, violent",
    },
    "cyberpunk_anime": {
        "display_name": "Cyberpunk Anime",
        "description": "Neon-lit cyberpunk anime aesthetic",
        "lora_file": "cyberpunk_anime.safetensors",
        "lora_weight": 0.72,
        "trigger_word": "cyberpunk anime style, neon lights, futuristic",
        "stage2_direction": "neon-lit urban environments, rain-slicked streets, holographic signs, dystopian cityscape",
        "negative_add": "countryside, nature, bright daylight",
    },
}

# ─────────────────────────────────────────────────────────────────────────
# Mood Grades — each entry drives Stage 2 direction + Stage 4 colour grade
# ─────────────────────────────────────────────────────────────────────────
MOOD_GRADES: Dict[str, Dict[str, Any]] = {
    "melancholic": {
        "stage2_direction": (
            "muted colour palette, overcast or rainy environments, cold blue-grey lighting, "
            "characters shown from behind or in profile, empty sparse backgrounds, "
            "fallen leaves, fog, long shadows, desaturated tones"
        ),
        "stage2_negative": "bright colors, vibrant, cheerful, sunny, smiling, saturated, energetic",
        "grade": {"saturation": 0.60, "brightness": 0.88, "color_shift": (0.85, 0.90, 1.10)},
    },
    "energetic": {
        "stage2_direction": "dynamic action, bold colours, strong contrast, motion blur, dramatic angles, bright lighting",
        "stage2_negative": "static, dull, grey, empty, quiet",
        "grade": {"saturation": 1.25, "brightness": 1.05, "color_shift": (1.05, 1.0, 0.92)},
    },
    "dark_intense": {
        "stage2_direction": "high contrast, deep shadows, dramatic underlighting, tension, storm clouds, threatening atmosphere",
        "stage2_negative": "soft, pastel, cheerful, bright, peaceful",
        "grade": {"saturation": 0.80, "brightness": 0.78, "color_shift": (0.90, 0.88, 1.05)},
    },
    "euphoric": {
        "stage2_direction": "golden hour lighting, warm tones, wide open spaces, freedom, bloom effects, uplifting vistas",
        "stage2_negative": "dark, gloomy, cramped, grey",
        "grade": {"saturation": 1.15, "brightness": 1.08, "color_shift": (1.08, 1.02, 0.90)},
    },
    "tense": {
        "stage2_direction": "tight framing, harsh shadows, red or amber emergency lighting, dramatic stillness before action",
        "stage2_negative": "relaxed, peaceful, open, soft",
        "grade": {"saturation": 0.90, "brightness": 0.85, "color_shift": (1.08, 0.90, 0.88)},
    },
    "neutral": {
        "stage2_direction": "",
        "stage2_negative": "",
        "grade": {"saturation": 1.0, "brightness": 1.0, "color_shift": (1.0, 1.0, 1.0)},
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
    Falls back to 'cyberpunk_anime' as a safe default.
    """
    mood_style_map = {
        "melancholic": "ghibli",
        "energetic": "mha",
        "euphoric": "mha",
        "dark_intense": "cyberpunk_anime",
        "tense": "cyberpunk_anime",
        "neutral": "none",
    }
    return mood_style_map.get(mood, "cyberpunk_anime")
