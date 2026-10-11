"""
AudioStory — Stage 4: Video Effects Engine

Takes generated frames, applies beat-driven visual effects to each frame,
and assembles the final video with the original audio.

Every effect is driven by the audio data from Stage 1 — all cuts, flashes,
and color grades map to the actual audio beats and drops.

Uses moviepy v2 API.
"""

import os
import time

import cv2
import numpy as np
from moviepy import ImageSequenceClip, AudioFileClip, CompositeVideoClip

from models.schemas import (
    GeneratedFrameSet, AudioAnalysisResult, FinalVideo, GeneratedFrame,
)
from utils.audio_utils import (
    get_energy_at_time, get_brightness_at_time, is_near_onset,
)

# ── Configuration ──────────────────────────────────────────────────────────────
TARGET_FPS = 30
OUTPUT_RESOLUTION = (1080, 1080)  # 1:1 for TikTok/Instagram


def build_video(
    frame_set: GeneratedFrameSet,
    audio_data: AudioAnalysisResult,
    audio_path: str,
    job_id: str,
    output_dir: str
) -> FinalVideo:
    """
    Main entry point: renders every frame with beat-synced effects,
    composes them into a video clip, and writes the final MP4.

    Parameters
    ----------
    frame_set : GeneratedFrameSet
        All generated images from Stage 3.
    audio_data : AudioAnalysisResult
        Audio fingerprint from Stage 1 (drives all effects).
    audio_path : str
        Path to the original audio file.
    job_id : str
        Unique job identifier.
    output_dir : str
        Base output directory.

    Returns
    -------
    FinalVideo
        Metadata about the exported video.
    """
    output_path = os.path.join(output_dir, job_id, "output.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    clips = []
    effects_log = set()

    total_frames = len(frame_set.frames)
    print(f"[Stage 4] Building video from {total_frames} frames at {TARGET_FPS}fps...")

    for i, frame in enumerate(frame_set.frames):
        print(f"[Stage 4] Rendering frame {frame.frame_id}/{total_frames} "
              f"({frame.type}) at t={frame.timestamp:.2f}s...")

        # Load and resize the image
        raw_image = load_and_resize(frame.image_path, OUTPUT_RESOLUTION)
        duration = frame.duration
        num_frames = max(1, int(duration * TARGET_FPS))

        # Get audio context at this frame's timestamp
        energy = get_energy_at_time(
            audio_data.rms_values, audio_data.rms_timestamps, frame.timestamp
        )
        brightness = get_brightness_at_time(
            audio_data.spectral_centroid_over_time,
            audio_data.rms_timestamps,
            frame.timestamp
        )
        is_onset = is_near_onset(audio_data.onset_times, frame.timestamp)
        is_first_of_scene = (i == 0 or frame_set.frames[i - 1].scene_id != frame.scene_id)

        frame_images = []

        for f_idx in range(num_frames):
            t = f_idx / TARGET_FPS
            current_frame = raw_image.copy()

            # ── Effect 1: Ken Burns (always active) ─────────────────────────
            current_frame = apply_ken_burns(current_frame, t, duration, energy)
            effects_log.add("ken_burns")

            # ── Effect 2: Beat Flash (first 3 frames of any beat cut) ───────
            if f_idx < 3 and frame.type.value == "KEYFRAME":
                flash_strength = min(0.7, energy * 0.9)
                current_frame = apply_flash(current_frame, flash_strength)
                effects_log.add("beat_flash")

            # ── Effect 3: Glitch (onset frames only) ────────────────────────
            if is_onset and f_idx < 5:
                current_frame = apply_glitch(current_frame, intensity=energy * 0.6)
                effects_log.add("glitch")

            # ── Effect 4: Color Grading (driven by spectral centroid) ────────
            current_frame = apply_color_grade(current_frame, energy, brightness)
            effects_log.add("color_grade")

            # ── Effect 5: Vignette (stronger at low energy moments) ──────────
            if energy < 0.35:
                current_frame = apply_vignette(current_frame, strength=0.6 - energy)
                effects_log.add("vignette")

            # ── Effect 6: Zoom Punch (at very high energy beats) ─────────────
            if is_onset and energy > 0.8 and f_idx < 8:
                zoom_factor = 1.0 + (0.06 * (1.0 - f_idx / 8))  # zoom then relax
                current_frame = apply_zoom(current_frame, zoom_factor)
                effects_log.add("zoom_punch")

            # Convert BGR → RGB for moviepy
            frame_images.append(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))

        # Create clip from this frame's rendered images (moviepy v2 API)
        clip = ImageSequenceClip(frame_images, fps=TARGET_FPS)
        clip = clip.with_start(frame.timestamp)
        clips.append(clip)

    print(f"[Stage 4] Assembling {len(clips)} clips into final video...")
    # Composite all clips — moviepy v2 accepts `size` parameter
    final_video = CompositeVideoClip(clips, size=OUTPUT_RESOLUTION)

    # Attach original audio
    audio_duration = min(30.0, audio_data.duration)
    audio = AudioFileClip(audio_path).subclipped(0, audio_duration)
    final_video = final_video.with_audio(audio)

    print(f"[Stage 4] Writing video to: {output_path}")
    final_video.write_videofile(
        output_path,
        fps=TARGET_FPS,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp_audio.m4a",
        remove_temp=True,
        logger=None  # set to 'bar' or None to suppress
    )

    return FinalVideo(
        job_id=job_id,
        output_path=output_path,
        duration=audio_data.duration,
        total_frames_used=total_frames,
        effects_applied=list(effects_log),
        audio_source=audio_path
    )


# ═══════════════════════════════════════════════════════
# EFFECTS LIBRARY
# ═══════════════════════════════════════════════════════

def apply_ken_burns(image, t, duration, energy):
    """
    Slow continuous zoom that makes still images feel alive.
    Energy drives zoom speed — high energy = faster zoom.
    """
    zoom_speed = 0.04 + (energy * 0.06)  # 4% to 10% zoom over duration
    progress = t / max(duration, 0.001)
    zoom = 1.0 + (zoom_speed * progress)

    h, w = image.shape[:2]
    new_w = int(w * zoom)
    new_h = int(h * zoom)
    resized = cv2.resize(image, (new_w, new_h))

    # Center crop back to original size
    x_off = (new_w - w) // 2
    y_off = (new_h - h) // 2
    return resized[y_off:y_off + h, x_off:x_off + w]


def apply_flash(image, strength):
    """
    White flash on beat cuts. Strength 0.0 = no flash, 1.0 = pure white.
    """
    white = np.ones_like(image, dtype=np.uint8) * 255
    return cv2.addWeighted(image, 1.0 - strength, white, strength, 0)


def apply_glitch(image, intensity):
    """
    Chromatic aberration + horizontal pixel slice displacement.
    Triggered only on onset/drop timestamps.
    """
    img = image.copy()
    shift = max(1, int(intensity * 25))

    # Chromatic aberration: shift R right, B left
    img[:, shift:, 2] = image[:, :-shift, 2]    # red channel right
    img[:, :-shift, 0] = image[:, shift:, 0]    # blue channel left

    # Random horizontal slice displacements
    num_slices = int(intensity * 8) + 2
    for _ in range(num_slices):
        y = np.random.randint(0, max(1, img.shape[0] - 20))
        slice_h = np.random.randint(2, min(18, img.shape[0] - y))
        slice_shift = np.random.randint(-35, 35)
        img[y:y + slice_h] = np.roll(img[y:y + slice_h], slice_shift, axis=1)

    return img


def apply_color_grade(image, energy, brightness):
    """
    Warm color grade for bright/energetic audio.
    Cool color grade for dark/heavy audio.
    Saturation boost at high energy moments.
    """
    img = image.astype(np.float32)

    if brightness > 0.55:
        # Warm grade: boost reds, reduce blues slightly
        img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 + energy * 0.35), 0, 255)  # R
        img[:, :, 0] = np.clip(img[:, :, 0] * (1.0 - energy * 0.15), 0, 255)  # B
    else:
        # Cool grade: boost blues, reduce reds slightly
        img[:, :, 0] = np.clip(img[:, :, 0] * (1.0 + energy * 0.35), 0, 255)  # B
        img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 - energy * 0.15), 0, 255)  # R

    # Saturation boost at high energy
    if energy > 0.6:
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + energy * 0.4), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)


def apply_vignette(image, strength):
    """
    Dark oval vignette that pulls the eye inward.
    Used at low energy / quiet moments.
    """
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    mask = np.clip(1.0 - dist * strength, 0, 1)
    mask = mask[:, :, np.newaxis]
    return (image * mask).astype(np.uint8)


def apply_zoom(image, zoom_factor):
    """
    Instantaneous zoom punch effect for hard drops.
    Zooms in and then relaxes over subsequent frames.
    """
    h, w = image.shape[:2]
    new_w = int(w * zoom_factor)
    new_h = int(h * zoom_factor)
    resized = cv2.resize(image, (new_w, new_h))
    x_off = (new_w - w) // 2
    y_off = (new_h - h) // 2
    return resized[y_off:y_off + h, x_off:x_off + w]


# ──────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────

def load_and_resize(image_path: str, target_size: tuple) -> np.ndarray:
    """
    Loads an image and resizes it to target_size (width, height).
    Returns a BGR numpy array for OpenCV processing.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load frame image: {image_path}")
    return cv2.resize(img, target_size)


if __name__ == "__main__":
    print("Stage 4 — Video Effects Engine")
    print("This module is called by the pipeline orchestrator.")
    print("Usage: from stages.stage4_video import build_video")
