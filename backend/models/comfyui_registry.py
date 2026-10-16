"""
AudioStory — ComfyUI Checkpoint Registry

Defines all available ComfyUI checkpoint models. Each entry maps to a .safetensors
file in the ComfyUI models/Stable-diffusion/ directory and provides optimal
generation parameters that the frontend auto-fills when the user selects a model.

Follows the same pattern as lora_registry.py.
"""

default_model_key = "meinamix"

COMFYUI_MODELS = {
    "meinamix": {
        "display_name": "MeinaMix V12",
        "filename": "meinamix_v12Final.safetensors",
        "optimal_steps": 20,
        "optimal_guidance": 7.0,
        "optimal_denoise": 0.75,
        "optimal_denoise_continuation": 0.45,
        "optimal_width": 768,
        "optimal_height": 768,
        "optimal_sampler": "euler",
        "optimal_scheduler": "normal",
        "notes": "Anime/manga style. Excellent character consistency. SD 1.5 based."
    },
    "sd15": {
        "display_name": "Stable Diffusion 1.5",
        "filename": "v1-5-pruned-emaonly-fp16.safetensors",
        "optimal_steps": 20,
        "optimal_guidance": 7.5,
        "optimal_denoise": 0.70,
        "optimal_denoise_continuation": 0.45,
        "optimal_width": 768,
        "optimal_height": 768,
        "optimal_sampler": "euler",
        "optimal_scheduler": "normal",
        "notes": "Base SD 1.5 model. Use with LoRA styles for consistent output."
    },
}


def get_model(model_key: str) -> dict:
    """Returns the model configuration dictionary for a given model key."""
    if model_key not in COMFYUI_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(COMFYUI_MODELS.keys())}")
    return COMFYUI_MODELS[model_key]


def list_models() -> dict:
    """
    Returns a simplified list of models for the API / frontend.
    Each entry includes the optimal generation parameters so the frontend
    can auto-fill the ComfyUI settings panel.
    """
    return {
        key: {
            "display_name": m["display_name"],
            "filename": m["filename"],
            "optimal_steps": m["optimal_steps"],
            "optimal_guidance": m["optimal_guidance"],
            "optimal_denoise": m["optimal_denoise"],
            "optimal_denoise_continuation": m["optimal_denoise_continuation"],
            "optimal_width": m["optimal_width"],
            "optimal_height": m["optimal_height"],
            "optimal_sampler": m["optimal_sampler"],
            "optimal_scheduler": m["optimal_scheduler"],
            "notes": m["notes"],
            "is_default": key == default_model_key,
        }
        for key, m in COMFYUI_MODELS.items()
    }
