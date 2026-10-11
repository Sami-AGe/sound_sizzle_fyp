"""
AudioStory — LoRA Style Registry

Defines all available visual styles. Each entry maps to a pre-trained LoRA file
that was trained once on Colab and saved to the loras/ directory.

For initial development (HF API only, no local .safetensors), the trigger_word
and style description are embedded directly in generation prompts.
"""

LORA_STYLES = {
    "mha_anime": {
        "display_name": "MHA Anime",
        "description": "Shonen anime style inspired by My Hero Academia",
        "lora_file": "loras/mha_anime.safetensors",
        "trigger_word": "mha style, anime, shonen, vibrant colors, hero costume, dynamic pose",
        "base_model": "stabilityai/stable-diffusion-2-1",
        "preview_image": "previews/mha_anime.jpg",
        "recommended_moods": ["energetic_euphoric", "intense_aggressive", "uplifting_heroic"]
    },
    "cyberpunk": {
        "display_name": "Cyberpunk Neon",
        "description": "Dark neon-lit cyberpunk aesthetic",
        "lora_file": "loras/cyberpunk.safetensors",
        "trigger_word": "cyberpunk style, neon lights, dark city, futuristic, holograms, rain",
        "base_model": "stabilityai/stable-diffusion-2-1",
        "preview_image": "previews/cyberpunk.jpg",
        "recommended_moods": ["dark_powerful", "melancholic_lonely", "chaotic_anxious"]
    },
    "lofi_pastel": {
        "display_name": "Lofi Pastel",
        "description": "Soft pastel tones, lofi anime illustration",
        "lora_file": "loras/lofi_pastel.safetensors",
        "trigger_word": "lofi style, soft pastel, warm illustration, gentle, cozy, dreamy",
        "base_model": "stabilityai/stable-diffusion-2-1",
        "preview_image": "previews/lofi_pastel.jpg",
        "recommended_moods": ["peaceful_dreamy", "melancholic_lonely", "neutral_cinematic"]
    },
    "dark_fantasy": {
        "display_name": "Dark Fantasy",
        "description": "Epic dark fantasy illustration style",
        "lora_file": "loras/dark_fantasy.safetensors",
        "trigger_word": "dark fantasy art, epic illustration, cinematic, dramatic, medieval, armor",
        "base_model": "stabilityai/stable-diffusion-2-1",
        "preview_image": "previews/dark_fantasy.jpg",
        "recommended_moods": ["dark_powerful", "intense_aggressive", "uplifting_heroic"]
    },
    "minimalist": {
        "display_name": "Minimalist",
        "description": "Clean flat minimalist illustration",
        "lora_file": "loras/minimalist.safetensors",
        "trigger_word": "minimalist illustration, flat design, clean lines, simple, geometric, limited colors",
        "base_model": "stabilityai/stable-diffusion-2-1",
        "preview_image": "previews/minimalist.jpg",
        "recommended_moods": ["peaceful_dreamy", "neutral_cinematic"]
    }
}

def get_style(style_key: str) -> dict:
    """Returns the style configuration dictionary for a given style key."""
    if style_key not in LORA_STYLES:
        raise ValueError(f"Unknown style: {style_key}. Available: {list(LORA_STYLES.keys())}")
    return LORA_STYLES[style_key]

def get_recommended_style(mood: str) -> str:
    """
    Returns the best style key for a given audio mood.
    Falls back to 'cyberpunk' as a safe default.
    """
    for key, style in LORA_STYLES.items():
        if mood in style["recommended_moods"]:
            return key
    return "cyberpunk"

def list_styles() -> dict:
    """
    Returns a simplified list of styles for the API / frontend.
    """
    return {
        key: {
            "display_name": s["display_name"],
            "description": s["description"],
            "preview_image": s["preview_image"]
        }
        for key, s in LORA_STYLES.items()
    }
