"""
Recovery script: picks up where a pipeline run was interrupted.

Usage:
    cd backend && python3 recover_job.py <job_id> <audio_path>

Loads the existing storyboard.json and audio_data.json from temp/<job_id>/,
generates any missing frames, creates frameset.json, and runs Stage 4 (video assembly).

This is a one-off recovery tool, not part of the regular pipeline.
"""
import sys
import os
import json
import time

# Load .env before importing backend modules
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stages.stage3_images import (
    client, NEGATIVE_PROMPT, TEXT_TO_IMAGE_MODEL,
    resolve_prompt_tokens, calculate_frame_duration,
    IMAGE_WIDTH, IMAGE_HEIGHT, GEN_STEPS, GUIDANCE_SCALE, FRAME_DELAY,
    USE_COMFYUI, COMFYUI_MODEL,
)
from PIL import Image  # for placeholder images
from stages.stage4_video import build_video
from models.schemas import (
    Storyboard, GeneratedFrame, GeneratedFrameSet, FrameType,
    AudioAnalysisResult,
)

if len(sys.argv) < 3:
    print("Usage: python3 recover_job.py <job_id> <audio_path>")
    print("  e.g. python3 recover_job.py 3d2b50fe ../temp/upload_a92ba8f9.wav")
    sys.exit(1)

job_id = sys.argv[1]
audio_path = sys.argv[2]
temp_dir = os.environ.get("OUTPUT_DIR", "./temp")
job_dir = os.path.join(temp_dir, job_id)
frames_dir = os.path.join(job_dir, "frames")

# ── Load existing artifacts ──────────────────────────────────────────────────────
with open(os.path.join(job_dir, "audio_data.json")) as f:
    audio_data = AudioAnalysisResult(**json.load(f))

with open(os.path.join(job_dir, "storyboard.json")) as f:
    storyboard = Storyboard(**json.load(f))

print(f"[Recovery] Job: {job_id}")
print(f"[Recovery] Audio: {audio_path}")
print(f"[Recovery] Storyboard: {storyboard.total_scenes} scenes, mode={storyboard.generation_mode}")
print(f"[Recovery] Provider: {'ComfyUI (' + COMFYUI_MODEL + ')' if USE_COMFYUI else 'HuggingFace Inference API'}")

# ── Reconcile frames: generate any missing ones ─────────────────────────────────
existing_frames = sorted(os.listdir(frames_dir)) if os.path.exists(frames_dir) else []
existing_count = len([f for f in existing_frames if f.startswith("frame_")])
total_expected = sum(len(s.frames) for s in storyboard.scenes)

print(f"[Recovery] Existing frames: {existing_count}/{total_expected}")

if existing_count >= total_expected:
    print("[Recovery] All frames present. Skipping Stage 3.")
    # We need to reconstruct the GeneratedFrameSet from existing data
    # Load frameset.json if it exists, otherwise rebuild from storyboard
    frameset_path = os.path.join(job_dir, "frameset.json")
    if os.path.exists(frameset_path):
        with open(frameset_path) as f:
            frame_set = GeneratedFrameSet(**json.load(f))
        print("[Recovery] Loaded existing frameset.json")
    else:
        print("[Recovery] frameset.json missing — rebuilding GeneratedFrameSet from storyboard + frames")
        generated_frames = []
        frame_progress = 0
        for scene in storyboard.scenes:
            for frame_directive in scene.frames:
                frame_progress += 1
                frame_path = os.path.join(frames_dir, f"frame_{frame_progress:03d}.png")
                duration = calculate_frame_duration(frame_directive, scene.frames, scene.end_time)
                # Determine prompt used
                if storyboard.generation_mode == "fast":
                    resolved = resolve_prompt_tokens(frame_directive.full_prompt, storyboard.character_anchor)
                    style_name = storyboard.style_anchor or "unknown"
                else:
                    resolved = frame_directive.full_prompt or ""
                    style_name = storyboard.subject_description or "unknown"
                generated_frames.append(GeneratedFrame(
                    frame_id=frame_directive.frame_id,
                    timestamp=frame_directive.timestamp,
                    type=frame_directive.type,
                    image_path=frame_path,
                    duration=duration,
                    scene_id=scene.scene_id,
                    prompt_used=resolved,
                    strength_used=None if storyboard.generation_mode.value == "fast" else (0.82 if frame_directive.type == FrameType.KEYFRAME else (frame_directive.strength or 0.22)),
                ))
        frame_set = GeneratedFrameSet(
            job_id=job_id,
            style_name=style_name,
            reference_image_path=os.path.join(frames_dir, "reference.png") if storyboard.generation_mode.value == "quality" else None,
            frames=generated_frames,
            total_frames=len(generated_frames),
            generation_mode=storyboard.generation_mode.value,
        )
else:
    # ── Generate missing frames ────────────────────────────────────────────────
    anchor = storyboard.character_anchor
    import random as _random

    # Resolution params — check for TEST_WIDTH/TEST_HEIGHT overrides
    width = int(os.environ.get("TEST_WIDTH", IMAGE_WIDTH))
    height = int(os.environ.get("TEST_HEIGHT", IMAGE_HEIGHT))
    steps = int(os.environ.get("TEST_STEPS", GEN_STEPS))
    guidance = float(os.environ.get("TEST_GUIDANCE", GUIDANCE_SCALE))

    mode = storyboard.generation_mode.value if hasattr(storyboard.generation_mode, 'value') else storyboard.generation_mode
    if mode == "fast":
        style_name = storyboard.style_anchor or "unknown"
    else:
        style_name = "unknown"

    generated_frames = []
    frame_progress = 0

    for scene in storyboard.scenes:
        scene_seed = scene.scene_seed
        for frame_directive in scene.frames:
            frame_progress += 1
            expected_path = os.path.join(frames_dir, f"frame_{frame_progress:03d}.png")

            if os.path.exists(expected_path):
                print(f"[Recovery] Frame {frame_progress}/{total_expected} already exists — skipping")
                duration = calculate_frame_duration(frame_directive, scene.frames, scene.end_time)
                if mode == "fast":
                    resolved = resolve_prompt_tokens(frame_directive.full_prompt, anchor)
                    generated_frames.append(GeneratedFrame(
                        frame_id=frame_directive.frame_id,
                        timestamp=frame_directive.timestamp,
                        type=frame_directive.type,
                        image_path=expected_path,
                        duration=duration,
                        scene_id=scene.scene_id,
                        prompt_used=resolved,
                        strength_used=None,
                    ))
                else:
                    modifier = frame_directive.modifier_prompt or ""
                    generated_frames.append(GeneratedFrame(
                        frame_id=frame_directive.frame_id,
                        timestamp=frame_directive.timestamp,
                        type=frame_directive.type,
                        image_path=expected_path,
                        duration=duration,
                        scene_id=scene.scene_id,
                        prompt_used=modifier or frame_directive.full_prompt,
                        strength_used=0.82 if frame_directive.type == FrameType.KEYFRAME else (frame_directive.strength or 0.22),
                    ))
                continue

            # ── Generate the missing frame ──────────────────────────────────────
            print(f"[Recovery] Generating missing frame {frame_progress}/{total_expected} "
                  f"({frame_directive.type} @ t={frame_directive.timestamp:.2f}s, seed={scene_seed})")

            if mode == "fast":
                resolved_prompt = resolve_prompt_tokens(frame_directive.full_prompt, anchor)
                try:
                    generated_image = client.text_to_image(
                        prompt=resolved_prompt,
                        model=TEXT_TO_IMAGE_MODEL,
                        negative_prompt=NEGATIVE_PROMPT,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        seed=scene_seed,
                    )
                    generated_image.save(expected_path)
                    print(f"[Recovery]   ✓ Saved: {expected_path} ({os.path.getsize(expected_path)} bytes)")
                except Exception as e:
                    print(f"[Recovery]   ◸ FAILED: {e}")
                    from PIL import Image
                    placeholder = Image.new('RGB', (width, height), color=(50, 50, 50))
                    placeholder.save(expected_path)
                    print(f"[Recovery]   Created placeholder ({width}x{height})")

                duration = calculate_frame_duration(frame_directive, scene.frames, scene.end_time)
                generated_frames.append(GeneratedFrame(
                    frame_id=frame_directive.frame_id,
                    timestamp=frame_directive.timestamp,
                    type=frame_directive.type,
                    image_path=expected_path,
                    duration=duration,
                    scene_id=scene.scene_id,
                    prompt_used=resolved_prompt,
                    strength_used=None,
                ))
            else:
                # Quality mode recovery — simpler, just generate from scratch
                # (This recovery is mainly for Fast mode)
                print(f"[Recovery]   Quality mode frame — generating reference-based image")
                from PIL import Image
                prompt = frame_directive.full_prompt or ""
                try:
                    generated_image = client.text_to_image(
                        prompt=prompt,
                        model=TEXT_TO_IMAGE_MODEL,
                        negative_prompt=NEGATIVE_PROMPT,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        seed=scene_seed,
                    )
                    generated_image.save(expected_path)
                except Exception as e:
                    print(f"[Recovery]   ◸ FAILED: {e}")
                    placeholder = Image.new('RGB', (width, height), color=(50, 50, 50))
                    placeholder.save(expected_path)

                duration = calculate_frame_duration(frame_directive, scene.frames, scene.end_time)
                generated_frames.append(GeneratedFrame(
                    frame_id=frame_directive.frame_id,
                    timestamp=frame_directive.timestamp,
                    type=frame_directive.type,
                    image_path=expected_path,
                    duration=duration,
                    scene_id=scene.scene_id,
                    prompt_used=prompt,
                    strength_used=None,
                ))

            time.sleep(FRAME_DELAY)

    frame_set = GeneratedFrameSet(
        job_id=job_id,
        style_name=style_name,
        reference_image_path=None if mode == "fast" else os.path.join(frames_dir, "reference.png"),
        frames=generated_frames,
        total_frames=len(generated_frames),
        generation_mode=mode,
    )

# ── Save frameset metadata ──────────────────────────────────────────────────────
frameset_path = os.path.join(job_dir, "frameset.json")
from stages.stage3_images import save_frameset_metadata
save_frameset_metadata(frame_set, frameset_path)
print(f"[Recovery] frameset.json saved: {frameset_path} ({os.path.getsize(frameset_path)} bytes)")

# ── Run Stage 4: Video Assembly ─────────────────────────────────────────────────
print(f"\n[Recovery] Stage 4: Building video from {frame_set.total_frames} frames...")
try:
    final_video = build_video(
        frame_set=frame_set,
        audio_data=audio_data,
        audio_path=audio_path,
        job_id=job_id,
        output_dir=temp_dir,
    )
    print(f"\n[Recovery] ✅ VIDEO COMPLETE")
    print(f"  Output: {final_video.output_path}")
    print(f"  Size: {os.path.getsize(final_video.output_path)} bytes ({os.path.getsize(final_video.output_path)/1024/1024:.1f} MB)")
    print(f"  Effects: {', '.join(final_video.effects_applied)}")
    print(f"  Frames used: {final_video.total_frames_used}")
except Exception as e:
    print(f"\n[Recovery] ❌ Stage 4 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n[Recovery] Done. Video available at: {final_video.output_path}")
