"""
AudioStory — Image Utility Functions

Helper functions for loading, resizing, converting, and saving images
throughout the pipeline.
"""

import os
import numpy as np
import cv2
from PIL import Image


def load_image_pil(path: str) -> Image.Image:
    """Loads an image file as a PIL Image."""
    return Image.open(path).convert("RGB")


def load_image_cv2(path: str) -> np.ndarray:
    """
    Loads an image file as a NumPy array in BGR format (OpenCV convention).
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def resize_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resizes an OpenCV image to the target (width, height)."""
    return cv2.resize(image, target_size)


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Converts a PIL Image (RGB) to OpenCV format (BGR)."""
    return np.array(image)[:, :, ::-1]


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Converts an OpenCV image (BGR) to PIL Image (RGB)."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def ensure_dir(path: str) -> None:
    """Creates directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_image_pil(image: Image.Image, path: str) -> None:
    """Saves a PIL Image to the given path, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


def save_image_cv2(image: np.ndarray, path: str) -> None:
    """Saves a NumPy/OpenCV image to the given path, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def overlay_transparent(background: np.ndarray, overlay: np.ndarray,
                        x: int, y: int, overlay_size: tuple = None) -> np.ndarray:
    """
    Overlays a transparent PNG onto a background image.
    Used for applying glitch artifacts and overlays.
    """
    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)

    if overlay.shape[2] == 4:
        mask = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y + overlay.shape[0], x:x + overlay.shape[1], c] = (
                background[y:y + overlay.shape[0], x:x + overlay.shape[1], c] * (1 - mask)
                + overlay[:, :, c] * mask
            )
    return background


def to_white_flash(image: np.ndarray, strength: float) -> np.ndarray:
    """
    Returns a white-flashed version of the image.
    strength=1.0 → pure white, strength=0.0 → original.
    """
    white = np.ones_like(image, dtype=np.uint8) * 255
    return cv2.addWeighted(image, 1.0 - strength, white, strength, 0)
