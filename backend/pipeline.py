"""
AudioStory — Pipeline Orchestrator

Single function that calls all four stages in sequence:
  Stage 1: Audio Analysis (librosa + Whisper)
  Stage 2: Storyboard Generation (Groq LLM)
  Stage 3: Image Generation (HuggingFace Inference API)
  Stage 4: Video Effects + Assembly (OpenCV + MoviePy)

Usage:
    from backend.pipeline import run_pipeline
    result = run_pipeline("my_audio.mp3", "mha_anime")
"""

import os
import uuid
import json
import time

from models.schemas import AudioAnalysisResult
from models.lora_registry import get_style
from stages.stage1_audio import analyze_audio
from stages.stage2_storyboard import generate_storyboard
from stages.stage3_images import (
    generate_all_frames, save_frameset_metadata,
    USE_COMFYUI, COMFYUI_MODEL,
)
from stages.stage4_video import build_video
from job_manager import update_job, log_progress, is_cancelled

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./temp")
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")


def run_pipeline(
    audio_path: str,
    style_key: str,
    whisper_model_size: str = WHISPER_MODEL_SIZE,
    max_frames: int = None,
    generation_mode: str = None,
    job_id: str = None,
    progress_job_id: str = None,
    gen_params: dict = None
) -> dict:
    """
    Runs the complete AudioStory pipeline end to end.

    Parameters
    ----------
    audio_path : str
        Path to the input audio file (max 30 seconds).
    style_key : str
        Key from lora_registry (e.g. "mha_anime", "cyberpunk").
    whisper_model_size : str
        Whisper model size for transcription.
    max_frames : int, optional
        Target total frame count for Stage 2/3. When provided, the
        storyboard LLM will aim for this many frames. Defaults to
        the MAX_FRAMES env var if set.
    generation_mode : str, optional
        "fast" (default) or "quality".
    job_id : str, optional
        Unique job identifier for this run. If None, one is generated.
    progress_job_id : str, optional
        Job ID for the job_manager (for frontend real-time progress).
        If None, no progress updates are sent to the job manager.
    gen_params : dict, optional
        Per-request generation parameter overrides (steps, guidance_scale,
        width, height, denoise, seed). Applied in Stage 3.

    Returns
    -------
    dict
        Summary with job_id, output_path, story, and stats.
    """
    # Read defaults from environment
    if max_frames is None:
        max_frames = os.environ.get("MAX_FRAMES")
        if max_frames:
            max_frames = int(max_frames)
    if generation_mode is None:
        generation_mode = os.environ.get("GENERATION_MODE", "fast")

    if job_id is None:
        job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    if progress_job_id:
        update_job(progress_job_id, status="running", current_stage=0)

    log_progress(progress_job_id, f"JOB {job_id} STARTED", "success")
    log_progress(progress_job_id, f"Audio: {audio_path}", "info")
    log_progress(progress_job_id, f"Style: {style_key} ({get_style(style_key)['display_name']})", "info")
    log_progress(progress_job_id, f"Mode: {generation_mode}", "info")
    log_progress(progress_job_id, f"Max frames: {max_frames}", "info")

    print(f"\n{'='*50}")
    print(f"JOB {job_id} STARTED")
    print(f"Audio: {audio_path}")
    print(f"Style: {style_key} ({get_style(style_key)['display_name']})")
    print(f"Mode: {generation_mode}")
    print(f"Max frames: {max_frames}")
    print(f"{'='*50}\n")

    style = get_style(style_key)

    try:
        # ── STAGE 1: Audio Analysis ──────────────────────────────────────────
        log_progress(progress_job_id, "[Stage 1] Analyzing audio with librosa + Whisper...", "info")
        if progress_job_id:
            update_job(progress_job_id, current_stage=1, stage_progress_msg="Analyzing audio")
        t0 = time.time()
        audio_data = analyze_audio(
            audio_path=audio_path,
            whisper_model_size=whisper_model_size,
            save_path=os.path.join(job_dir, "audio_data.json")
        )
        t1 = time.time()
        log_progress(progress_job_id,
            f"[Stage 1] Complete in {t1-t0:.1f}s. Mood: {audio_data.detected_mood}", "success")

        if is_cancelled(progress_job_id):
            log_progress(progress_job_id, "[Pipeline] Cancelled before Stage 2", "warn")
            update_job(progress_job_id, status="cancelled", end_time=time.time())
            return {"job_id": job_id, "status": "cancelled"}

        # ── STAGE 2: Storyboard Generation ──────────────────────────────────
        log_progress(progress_job_id, f"[Stage 2] Generating storyboard via Groq (mode={generation_mode})...", "info")
        if progress_job_id:
            update_job(progress_job_id, current_stage=2, stage_progress_msg="Generating storyboard")
        t2 = time.time()
        storyboard = generate_storyboard(
            audio_data=audio_data,
            style_name=style["display_name"],
            style_description=style["description"],
            style_trigger_word=style["trigger_word"],
            save_path=os.path.join(job_dir, "storyboard.json"),
            max_frames=max_frames,
            generation_mode=generation_mode
        )
        t3 = time.time()
        total_frames_in_story = sum(len(s.frames) for s in storyboard.scenes)
        log_progress(progress_job_id,
            f"[Stage 2] Complete in {t3-t2:.1f}s. {storyboard.total_scenes} scenes, {total_frames_in_story} frames", "success")

        if is_cancelled(progress_job_id):
            log_progress(progress_job_id, "[Pipeline] Cancelled before Stage 3", "warn")
            update_job(progress_job_id, status="cancelled", end_time=time.time())
            return {"job_id": job_id, "status": "cancelled"}

        # ── STAGE 3: Image Generation ────────────────────────────────────────
        if USE_COMFYUI:
            log_progress(progress_job_id, f"[Stage 3] Generating images via ComfyUI ({COMFYUI_MODEL})...", "info")
        else:
            log_progress(progress_job_id, "[Stage 3] Generating images via HuggingFace API...", "info")
        if progress_job_id:
            update_job(progress_job_id, current_stage=3, stage_progress_msg="Generating images")
        t4 = time.time()
        frame_set = generate_all_frames(
            storyboard=storyboard,
            style_name=style["display_name"],
            style_trigger_word=style["trigger_word"],
            job_id=job_id,
            output_dir=OUTPUT_DIR,
            progress_job_id=progress_job_id,
            gen_params=gen_params
        )
        save_frameset_metadata(frame_set, os.path.join(job_dir, "frameset.json"))
        t5 = time.time()
        log_progress(progress_job_id,
            f"[Stage 3] Complete in {t5-t4:.1f}s. {frame_set.total_frames} frames generated", "success")

        if is_cancelled(progress_job_id):
            log_progress(progress_job_id, "[Pipeline] Cancelled before Stage 4", "warn")
            update_job(progress_job_id, status="cancelled", end_time=time.time())
            return {"job_id": job_id, "status": "cancelled"}

        # ── STAGE 4: Video Assembly ──────────────────────────────────────────
        log_progress(progress_job_id, "[Stage 4] Building video with beat-synced effects...", "info")
        if progress_job_id:
            update_job(progress_job_id, current_stage=4, stage_progress_msg="Building video")
        t6 = time.time()
        final_video = build_video(
            frame_set=frame_set,
            audio_data=audio_data,
            audio_path=audio_path,
            job_id=job_id,
            output_dir=OUTPUT_DIR
        )
        t7 = time.time()
        log_progress(progress_job_id,
            f"[Stage 4] Complete in {t7-t6:.1f}s", "success")

        log_progress(progress_job_id, f"JOB {job_id} COMPLETE — Total time: {t7-t0:.1f}s", "success")
        log_progress(progress_job_id, f"Output: {final_video.output_path}", "info")
        log_progress(progress_job_id,
            f"Effects applied: {', '.join(final_video.effects_applied)}", "info")

        print(f"\n{'='*50}")
        print(f"JOB {job_id} COMPLETE")
        print(f"Total time: {t7-t0:.1f}s")
        print(f"Output: {final_video.output_path}")
        print(f"Effects applied: {', '.join(final_video.effects_applied)}")
        print(f"{'='*50}\n")

        result = {
            "job_id": job_id,
            "output_path": final_video.output_path,
            "story": storyboard.main_story,
            "style": style["display_name"],
            "mood_detected": audio_data.detected_mood,
            "total_scenes": storyboard.total_scenes,
            "total_frames": frame_set.total_frames,
            "max_frames_target": max_frames,
            "effects_applied": final_video.effects_applied,
            "processing_time_seconds": round(t7 - t0, 1),
            "generation_mode": generation_mode,
        }

        if progress_job_id:
            update_job(progress_job_id, status="completed", end_time=time.time(), result=result)

        return result

    except Exception as e:
        log_progress(progress_job_id, f"[Pipeline] ERROR: {e}", "error")
        if progress_job_id:
            update_job(progress_job_id, status="failed", end_time=time.time(), error=str(e))
        raise


if __name__ == "__main__":
    import sys
    from models.lora_registry import LORA_STYLES

    if len(sys.argv) < 3:
        print("Usage: python -m backend.pipeline <audio_path> <style_key>")
        print("Styles:", list(LORA_STYLES.keys()))
        sys.exit(1)

    max_frames = int(os.environ.get("MAX_FRAMES", "0")) or None
    generation_mode = os.environ.get("GENERATION_MODE", "fast")
    result = run_pipeline(
        sys.argv[1], sys.argv[2],
        max_frames=max_frames,
        generation_mode=generation_mode
    )
    print(json.dumps(result, indent=2))
