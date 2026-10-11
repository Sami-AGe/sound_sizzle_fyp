"""
AudioStory — Stage 3: Image Generation Engine

Walks through the storyboard frame by frame and generates images.

v2 dual-mode: Two completely independent code paths.
  Fast mode:   EVERY frame → text_to_image (SD3-Medium, seed-locked per scene)
               Consistency via Character Anchor tokens + scene seed
  Quality mode: Reference image + img2img (FLUX.1-Kontext, pixel inheritance)

Both share the same generation parameters (768×768, 20 steps, 7.5 guidance).
"""

import os
import time
import json

from PIL import Image
from huggingface_hub import InferenceClient
from comfyui_client import ComfyUIClient, COMFYUI_MODEL
from job_manager import log_progress, is_cancelled
from models.schemas import (
    Storyboard, GeneratedFrame, GeneratedFrameSet, FrameType,
)

# ─────────────────────────────────────────────────────────────────────────
# Client selection: ComfyUI (local) or HuggingFace Inference API.
# Controlled by USE_COMFYUI env var. When ComfyUI is active, the
# ComfyUIClient exposes the same text_to_image()/image_to_image()
# interface as InferenceClient, so all downstream code is unchanged.
# ─────────────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
PROVIDER = os.environ.get("HF_PROVIDER", "replicate")
USE_COMFYUI = os.environ.get("USE_COMFYUI", "false").lower() == "true"

if USE_COMFYUI:
    print("[Stage 3] Using ComfyUI for image generation")
    client = ComfyUIClient()
else:
    print("[Stage 3] Using HuggingFace Inference API")
    client = InferenceClient(token=HF_TOKEN, provider=PROVIDER)

# Two-model strategy: SD3 Medium for text-to-image (reference),
# FLUX.1-Kontext-dev for image-to-image (keyframes + continuations in Quality mode)
TEXT_TO_IMAGE_MODEL = os.environ.get(
    "TEXT_TO_IMAGE_MODEL", "stabilityai/stable-diffusion-3-medium"
)
IMAGE_TO_IMAGE_MODEL = os.environ.get(
    "IMAGE_TO_IMAGE_MODEL", "black-forest-labs/FLUX.1-Kontext-dev"
)

# ─────────────────────────────────────────────────────────────────────────
# Generation parameters
# Lowered from 1024→768 and 30→20 steps to reduce cost/time while maintaining
# content faithfulness. Final video outputs at 1080×1080, so 768×768 is sufficient.
# Stylized diffusion at 20 steps produces faithful content without realism overhead.
# ─────────────────────────────────────────────────────────────────────────

IMAGE_WIDTH = 768
IMAGE_HEIGHT = 768
GEN_STEPS = 20
GUIDANCE_SCALE = 7.5


def build_negative_prompt() -> str:
    """Standard negative prompt for consistent quality.
    Actively bans photorealistic/3D realism to keep output stylized and reduce cost.
    """
    return (
        "bad anatomy, deformed, ugly, blurry, low quality, watermark, "
        "text, signature, extra limbs, mutated, distorted face, "
        "inconsistent character, "
        "photorealistic, photo, photograph, realistic, hyperrealistic, "
        "3d render, cgi, ultra detailed, 8k, 4k, hdr"
    )


# Module-level constant matching build_negative_prompt()
NEGATIVE_PROMPT = build_negative_prompt()


def resolve_prompt_tokens(raw_prompt: str, anchor) -> str:
    """
    Substitutes {character_anchor} and {style_anchor} tokens in Fast mode prompts.
    Called on every Fast mode frame prompt before sending to HF.

    The anchor is a CharacterAnchor Pydantic model (after Storyboard construction),
    so dot notation is used: anchor.full_anchor_text, anchor.style_anchor.
    """
    resolved = raw_prompt.replace("{character_anchor}", anchor.full_anchor_text)
    resolved = resolved.replace("{style_anchor}", anchor.style_anchor)
    return resolved


def calculate_frame_duration(
    frame_directive,
    all_scene_frames,
    scene_end_time: float
) -> float:
    """
    Duration = time until the next frame, or until scene ends.
    """
    idx = next(
        i for i, f in enumerate(all_scene_frames)
        if f.frame_id == frame_directive.frame_id
    )
    if idx + 1 < len(all_scene_frames):
        return all_scene_frames[idx + 1].timestamp - frame_directive.timestamp
    else:
        return scene_end_time - frame_directive.timestamp


def generate_all_frames_fast(
    storyboard: Storyboard,
    style_name: str,
    style_trigger_word: str,
    job_id: str,
    output_dir: str,
    progress_job_id: str = None,
    gen_params: dict = None
) -> GeneratedFrameSet:
    """
    Fast mode: ALL frames use text_to_image (SD3-Medium).
    Consistency achieved via Character Anchor tokens + scene seed locking.
    No img2img calls, no reference image.

    Note: style_name is not used in Fast mode (style_anchor from storyboard is used
    instead), but is accepted for signature parity with the Quality mode path and
    the generate_all_frames dispatcher.

    progress_job_id: If set, sends log entries to the job manager for frontend
                     real-time progress display.
    gen_params: Per-request overrides for steps, guidance_scale, width, height, seed.
                Falls back to module constants (GEN_STEPS, GUIDANCE_SCALE, etc.)
                when not provided.
    """
    # Resolve generation parameters: use overrides if provided, else defaults
    gen_params = gen_params or {}
    steps = gen_params.get("steps", GEN_STEPS)
    guidance = gen_params.get("guidance_scale", GUIDANCE_SCALE)
    width = gen_params.get("width", IMAGE_WIDTH)
    height = gen_params.get("height", IMAGE_HEIGHT)
    user_seed_override = gen_params.get("seed", None)  # None or -1 = random

    # Import random lazily to avoid module-level dependency
    import random as _random
    frames_dir = os.path.join(output_dir, job_id, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    anchor = storyboard.character_anchor
    generated_frames = []

    total_frames_count = sum(len(s.frames) for s in storyboard.scenes)
    frame_progress = 0

    log_progress(progress_job_id, f"[Stage 3 FAST] Starting image generation: {total_frames_count} frames", "info")

    _cancelled = False
    for scene in storyboard.scenes:
        if _cancelled:
            break
        # Determine seed: user override if provided (non-None, non -1), else scene_seed
        if user_seed_override is not None and user_seed_override != -1:
            frame_seed = user_seed_override
            log_progress(progress_job_id, f"[Stage 3] Scene {scene.scene_id} — {len(scene.frames)} frames (global seed={frame_seed})", "info")
        else:
            frame_seed = scene.scene_seed
            log_progress(progress_job_id, f"[Stage 3] Scene {scene.scene_id} — seed={frame_seed}, {len(scene.frames)} frames", "info")

        for frame_directive in scene.frames:
            if is_cancelled(progress_job_id):
                log_progress(progress_job_id, "[Stage 3] Generation cancelled by user", "warn")
                _cancelled = True
                break

            frame_counter = frame_directive.frame_id
            frame_progress += 1
            frame_path = os.path.join(frames_dir, f"frame_{frame_directive.frame_id:03d}.png")

            # Resolve {character_anchor} and {style_anchor} tokens
            resolved_prompt = resolve_prompt_tokens(frame_directive.full_prompt, anchor)

            log_progress(progress_job_id,
                f"[Stage 3] Generating frame {frame_counter}/{total_frames_count} "
                f"({frame_directive.type}) @ seed={frame_seed}", "info")

            try:
                generated_image = client.text_to_image(
                    prompt=resolved_prompt,
                    model=TEXT_TO_IMAGE_MODEL,
                    negative_prompt=NEGATIVE_PROMPT,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    seed=frame_seed      # Locked per scene for visual consistency,
                                        # or global seed override from frontend
                )

                generated_image.save(frame_path)

                duration = calculate_frame_duration(frame_directive, scene.frames, scene.end_time)
                generated_frames.append(GeneratedFrame(
                    frame_id=frame_directive.frame_id,
                    timestamp=frame_directive.timestamp,
                    type=frame_directive.type,
                    image_path=frame_path,
                    duration=duration,
                    scene_id=scene.scene_id,
                    prompt_used=resolved_prompt,
                    strength_used=None   # N/A for Fast mode
                ))

                log_progress(progress_job_id,
                    f"[Stage 3] Frame {frame_counter}/{total_frames_count} saved ✓ ({frame_directive.type})", "success")

            except Exception as e:
                log_progress(progress_job_id,
                    f"[Stage 3] ERROR: Frame {frame_counter} failed: {e}. Continuing with remaining frames.", "error")
                # Create a placeholder so downstream stages don't break
                placeholder = Image.new('RGB', (width, height), color=(50, 50, 50))
                placeholder.save(frame_path)
                duration = calculate_frame_duration(frame_directive, scene.frames, scene.end_time)
                generated_frames.append(GeneratedFrame(
                    frame_id=frame_directive.frame_id,
                    timestamp=frame_directive.timestamp,
                    type=frame_directive.type,
                    image_path=frame_path,
                    duration=duration,
                    scene_id=scene.scene_id,
                    prompt_used=resolved_prompt,
                    strength_used=None
                ))

            time.sleep(0.4)

    log_progress(progress_job_id, f"[Stage 3 FAST] Complete: {len(generated_frames)}/{total_frames_count} frames generated", "success")

    return GeneratedFrameSet(
        job_id=job_id,
        style_name=storyboard.style_anchor or "unknown",
        reference_image_path=None,   # Fast mode has no separate reference image
        frames=generated_frames,
        total_frames=len(generated_frames),
        generation_mode="fast"
    )


def generate_all_frames_quality(
    storyboard: Storyboard,
    style_name: str,
    style_trigger_word: str,
    job_id: str,
    output_dir: str,
    progress_job_id: str = None,
    gen_params: dict = None
) -> GeneratedFrameSet:
    """
    Quality mode: Reference image + img2img via FLUX.1-Kontext.
    Visual consistency achieved via pixel inheritance.

    progress_job_id: If set, sends log entries to the job manager for frontend
                     real-time progress display.
    gen_params: Per-request overrides for steps, guidance_scale, width, height, denoise, seed.
                Falls back to module constants (GEN_STEPS, GUIDANCE_SCALE, etc.)
                when not provided.
    """
    # Resolve generation parameters: use overrides if provided, else defaults
    gen_params = gen_params or {}
    steps = gen_params.get("steps", GEN_STEPS)
    guidance = gen_params.get("guidance_scale", GUIDANCE_SCALE)
    width = gen_params.get("width", IMAGE_WIDTH)
    height = gen_params.get("height", IMAGE_HEIGHT)
    denoise_override = gen_params.get("denoise", None)  # If set, overrides per-frame strength
    user_seed_override = gen_params.get("seed", None)  # None or -1 = random

    import random as _random

    frames_dir = os.path.join(output_dir, job_id, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    generated_frames = []

    # Generate reference image (used as base for all keyframes)
    log_progress(progress_job_id, "[Stage 3 QUALITY] Generating reference image...", "info")
    reference_prompt = storyboard.subject_reference_prompt
    try:
        reference_image = client.text_to_image(
            prompt=f"{reference_prompt}, {style_trigger_word}, illustration, stylized",
            model=TEXT_TO_IMAGE_MODEL,
            negative_prompt=NEGATIVE_PROMPT,
            width=width,
            height=height,
            num_inference_steps=steps,
            seed=_random.randint(0, 2**32-1) if not user_seed_override or user_seed_override == -1 else user_seed_override,
        )
    except Exception as e:
        log_progress(progress_job_id, f"[Stage 3] ERROR: Reference image failed: {e}", "error")
        reference_image = Image.new('RGB', (width, height), color=(50, 50, 50))

    reference_path = os.path.join(frames_dir, "reference.png")
    reference_image.save(reference_path)
    log_progress(progress_job_id, f"[Stage 3] Reference image saved: {reference_path}", "success")
    reference_image_data = reference_image  # Use in-memory image directly (already RGB from text_to_image)
    previous_image = reference_image_data

    total_frame_count = sum(len(s.frames) for s in storyboard.scenes)
    frame_counter = 0

    log_progress(progress_job_id, f"[Stage 3 QUALITY] Starting image generation: {total_frame_count} frames", "info")

    _cancelled = False
    for scene in storyboard.scenes:
        if _cancelled:
            break
        log_progress(progress_job_id, f"[Stage 3] Scene {scene.scene_id} — {len(scene.frames)} frames", "info")

        for frame_directive in scene.frames:
            if is_cancelled(progress_job_id):
                log_progress(progress_job_id, "[Stage 3] Generation cancelled by user", "warn")
                _cancelled = True
                break

            frame_counter += 1
            frame_num = frame_directive.frame_id
            frame_path = os.path.join(frames_dir, f"frame_{frame_num:03d}.png")

            log_progress(progress_job_id,
                f"[Stage 3] [{frame_counter}/{total_frame_count}] "
                f"Generating {frame_directive.type} frame at t={frame_directive.timestamp:.2f}s...", "info")

            try:
                if frame_directive.type == FrameType.KEYFRAME:
                    # Fresh generation but guided by reference image for consistency
                    full_prompt = (
                        f"{frame_directive.full_prompt}, "
                        f"{style_trigger_word}, "
                        f"{scene.color_palette}, "
                        f"illustration, stylized"
                    )
                    # Strength 0.82 for keyframes (heavy change), or override from frontend
                    strength = denoise_override if denoise_override is not None else 0.82

                    current_image = client.image_to_image(
                        image=reference_image_data,
                        model=IMAGE_TO_IMAGE_MODEL,
                        prompt=full_prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        strength=strength,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        seed=_random.randint(0, 2**32-1) if not user_seed_override or user_seed_override == -1 else user_seed_override,
                    )

                elif frame_directive.type == FrameType.CONTINUATION:
                    # Evolve the PREVIOUS frame slightly
                    modifier_prompt = (
                        f"{storyboard.subject_description}, "
                        f"{frame_directive.modifier_prompt}, "
                        f"{scene.color_palette}, "
                        f"{style_trigger_word}"
                    )
                    # Strength 0.22 for continuations (subtle change), or storyboard value, or override
                    strength = denoise_override if denoise_override is not None else (frame_directive.strength or 0.22)

                    current_image = client.image_to_image(
                        image=previous_image,
                        model=IMAGE_TO_IMAGE_MODEL,
                        prompt=modifier_prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        strength=strength,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        seed=_random.randint(0, 2**32-1) if not user_seed_override or user_seed_override == -1 else user_seed_override,
                    )

                else:
                    # Fallback: generate from scratch
                    current_image = client.text_to_image(
                        prompt=f"{frame_directive.full_prompt or frame_directive.modifier_prompt}, {style_trigger_word}",
                        model=TEXT_TO_IMAGE_MODEL,
                        negative_prompt=NEGATIVE_PROMPT,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        seed=_random.randint(0, 2**32-1) if not user_seed_override or user_seed_override == -1 else user_seed_override,
                    )

                current_image.save(frame_path)
                previous_image = current_image.copy()

                log_progress(progress_job_id,
                    f"[Stage 3] Frame {frame_counter}/{total_frame_count} saved ✓ ({frame_directive.type})", "success")

            except Exception as e:
                log_progress(progress_job_id,
                    f"[Stage 3] ERROR: Frame {frame_counter} failed: {e}. Continuing with remaining frames.", "error")
                # Create a placeholder so downstream stages don't break
                placeholder = Image.new('RGB', (width, height), color=(50, 50, 50))
                placeholder.save(frame_path)
                previous_image = placeholder.copy()

            # Calculate hold duration
            duration = calculate_frame_duration(
                frame_directive, scene.frames, scene.end_time
            )

            generated_frames.append(GeneratedFrame(
                frame_id=frame_directive.frame_id,
                timestamp=frame_directive.timestamp,
                type=frame_directive.type,
                image_path=frame_path,
                duration=max(0.01, duration),
                scene_id=scene.scene_id,
                prompt_used=(frame_directive.full_prompt or frame_directive.modifier_prompt),
                strength_used=(denoise_override if denoise_override is not None
                               else (0.82 if frame_directive.type == FrameType.KEYFRAME
                                     else (frame_directive.strength or 0.22)))
            ))

            # Rate limit courtesy pause
            time.sleep(0.5)

    log_progress(progress_job_id, f"[Stage 3 QUALITY] Complete: {len(generated_frames)}/{total_frame_count} frames generated", "success")

    return GeneratedFrameSet(
        job_id=job_id,
        style_name=style_name,
        reference_image_path=reference_path,
        frames=generated_frames,
        total_frames=len(generated_frames),
        generation_mode="quality"
    )


def generate_all_frames(
    storyboard: Storyboard,
    style_name: str,
    style_trigger_word: str,
    job_id: str,
    output_dir: str,
    progress_job_id: str = None,
    gen_params: dict = None
) -> GeneratedFrameSet:
    """
    Public entry point. Reads generation_mode from storyboard and dispatches
    to the appropriate code path. Called by pipeline.py.

    Parameters
    ----------
    storyboard : Storyboard
        The structured visual plan from Stage 2.
    style_name : str
        Display name of the visual style.
    style_trigger_word : str
        Trigger words to embed in generation prompts for style guidance.
    job_id : str
        Unique job identifier (used for frame directory).
    output_dir : str
        Base output directory.
    progress_job_id : str, optional
        Job ID for the job manager, if progress tracking is enabled.
    gen_params : dict, optional
        Per-request generation parameter overrides (steps, guidance_scale,
        width, height, denoise, seed). Applied in the generation code paths.

    Returns
    -------
    GeneratedFrameSet
        All generated frames with metadata.
    """
    mode = getattr(storyboard, 'generation_mode', None)
    if mode is None:
        mode = "fast"
    else:
        mode = mode.value if hasattr(mode, 'value') else mode

    if mode == "quality":
        return generate_all_frames_quality(
            storyboard, style_name, style_trigger_word, job_id, output_dir,
            progress_job_id, gen_params
        )
    else:
        return generate_all_frames_fast(
            storyboard, style_name, style_trigger_word, job_id, output_dir,
            progress_job_id, gen_params
        )


def save_frameset_metadata(frame_set: GeneratedFrameSet, save_path: str) -> None:
    """Saves the frame set metadata as JSON for inspection/debugging."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(frame_set.model_dump(), f, indent=2)


if __name__ == "__main__":
    import sys
    from stages.stage1_audio import analyze_audio
    from stages.stage2_storyboard import generate_storyboard
    from models.lora_registry import get_style

    if len(sys.argv) < 3:
        print("Usage: python -m backend.stages.stage3_images <audio_path> <style_key>")
        sys.exit(1)

    audio_file = sys.argv[1]
    style_key = sys.argv[2]
    style = get_style(style_key)

    # Run previous stages
    analysis = analyze_audio(audio_file, save_path=f"./temp/{os.path.basename(audio_file)}_analysis.json")

    storyboard = generate_storyboard(
        audio_data=analysis,
        style_name=style["display_name"],
        style_description=style["description"],
        style_trigger_word=style["trigger_word"],
        save_path=f"./temp/{os.path.basename(audio_file)}_storyboard.json"
    )

    import uuid
    job_id = str(uuid.uuid4())[:8]

    frame_set = generate_all_frames(
        storyboard=storyboard,
        style_name=style["display_name"],
        style_trigger_word=style["trigger_word"],
        job_id=job_id,
        output_dir="./temp"
    )

    save_frameset_metadata(frame_set, f"./temp/{job_id}_frameset.json")
    print(f"\n[Stage 3] Job {job_id} complete.")
    print(f"  Frames: {frame_set.total_frames}")
    print(f"  Generation mode: {frame_set.generation_mode}")
    print(f"  Reference: {frame_set.reference_image_path}")
    for f in frame_set.frames[:5]:
        print(f"  Frame {f.frame_id}: {f.type} @ {f.timestamp:.2f}s → {f.image_path}")
