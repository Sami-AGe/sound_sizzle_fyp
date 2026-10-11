"""
AudioStory — FastAPI Backend

Endpoints:
  GET  /styles          — List available visual styles
  POST /generate        — Run the full pipeline (audio → video)
  GET  /download/{job_id} — Download a completed video
  GET  /jobs/{job_id}   — Check job status / metadata
  POST /test/stage1     — Run only Stage 1 (audio analysis) for debugging
  POST /test/stage2     — Run Stage 1 + Stage 2 (audio + storyboard) for debugging
"""

import os
import sys
import uuid
import json
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add backend directory to Python path so flat imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from pipeline import run_pipeline
from models.lora_registry import LORA_STYLES, get_style, list_styles
from models.schemas import AudioAnalysisResult
from stages.stage1_audio import analyze_audio, determine_scene_count
from stages.stage2_storyboard import generate_storyboard
from job_manager import create_job, get_job, cancel_job, log_progress, update_job

app = FastAPI(title="AudioStory API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

TEMP_DIR = os.environ.get("OUTPUT_DIR", "./temp")
os.makedirs(TEMP_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"service": "AudioStory", "version": "1.0.0", "status": "running"}


@app.get("/styles")
def get_styles():
    """Lists all available visual styles with their display names and descriptions."""
    return list_styles()


@app.get("/config")
def get_config():
    """Returns the current backend configuration for the frontend to display."""
    from stages.stage3_images import USE_COMFYUI, COMFYUI_MODEL
    return {
        "use_comfyui": USE_COMFYUI,
        "comfyui_host": os.environ.get("COMFYUI_HOST", "http://127.0.0.1:8188"),
        "comfyui_model": COMFYUI_MODEL,
        "hf_provider": os.environ.get("HF_PROVIDER", "replicate"),
        "text_to_image_model": os.environ.get("TEXT_TO_IMAGE_MODEL", "stabilityai/stable-diffusion-3-medium"),
        "image_to_image_model": os.environ.get("IMAGE_TO_IMAGE_MODEL", "black-forest-labs/FLUX.1-Kontext-dev"),
        "generation_mode_default": os.environ.get("GENERATION_MODE", "fast"),
        "max_frames_default": int(os.environ.get("MAX_FRAMES", "15")),
        "whisper_model_size": os.environ.get("WHISPER_MODEL_SIZE", "base"),
    }


@app.post("/generate")
async def generate_video(
    audio: UploadFile = File(...),
    style: str = Form(...),
    max_frames: int = Form(None),
    mode: str = Form(None),        # "fast" or "quality"; None = read from .env
    steps: int = Form(None),       # Inference steps override
    guidance: float = Form(None),  # Guidance scale override
    width: int = Form(None),       # Image width override
    height: int = Form(None),      # Image height override
    denoise: float = Form(None),   # Strength/denoise override (Quality mode)
    seed: int = Form(None),        # Seed override (-1 = random)
):
    """
    Runs the complete AudioStory pipeline asynchronously.

    Uploads an audio file (max 30 seconds) and selects a style.
    Returns immediately with a job_id for polling progress via /jobs/{job_id}.

    Parameters:
    - audio: The audio file (mp3, wav, m4a, ogg)
    - style: Visual style key (e.g. "cyberpunk", "mha_anime")
    - max_frames: Optional target total frame count
    - mode: Optional generation mode: "fast" (default) or "quality"
    - steps: Override for inference steps (default from env/code)
    - guidance: Override for guidance scale
    - width: Override for image width
    - height: Override for image height
    - denoise: Override for img2img denoise/strength (Quality mode)
    - seed: Override for generation seed (-1 = random)

    Returns:
    - JSON with job_id, status, and configuration (poll /jobs/{job_id} for progress)
    """
    if mode and mode not in ("fast", "quality"):
        raise HTTPException(400, "mode must be 'fast' or 'quality'")

    # Validate audio type
    if not audio.filename.endswith((".mp3", ".wav", ".m4a", ".ogg")):
        raise HTTPException(400, "Unsupported audio format. Use mp3, wav, m4a, or ogg.")

    if style not in LORA_STYLES:
        raise HTTPException(400, f"Invalid style. Choose from: {list(LORA_STYLES.keys())}")

    # Save uploaded file
    upload_id = str(uuid.uuid4())[:8]
    upload_path = os.path.join(
        TEMP_DIR, f"upload_{upload_id}{os.path.splitext(audio.filename)[1]}"
    )

    with open(upload_path, "wb") as f:
        content = await audio.read()
        f.write(content)

    # Build generation params dict (only include non-None overrides)
    gen_params = {}
    if steps is not None: gen_params["steps"] = steps
    if guidance is not None: gen_params["guidance_scale"] = guidance
    if width is not None: gen_params["width"] = width
    if height is not None: gen_params["height"] = height
    if denoise is not None: gen_params["denoise"] = denoise
    if seed is not None: gen_params["seed"] = seed

    # Create job for progress tracking
    job_id = str(uuid.uuid4())[:8]
    create_job(job_id)

    # Store gen_params in job state so they're accessible during the run
    if gen_params:
        update_job(job_id, stage_progress_msg=json.dumps({"gen_params": gen_params}))
    log_progress(job_id, f"Pipeline started: job_id={job_id}, style={style}, mode={mode}, max_frames={max_frames}", "info")

    # Run pipeline in background thread so the response returns immediately
    import threading

    def run_pipeline_background():
        try:
            run_pipeline(
                audio_path=upload_path,
                style_key=style,
                max_frames=max_frames,
                generation_mode=mode,
                progress_job_id=job_id,
                gen_params=gen_params if gen_params else None,
            )
        except Exception as e:
            log_progress(job_id, f"Pipeline failed: {e}", "error")
        finally:
            if os.path.exists(upload_path):
                os.remove(upload_path)

    thread = threading.Thread(target=run_pipeline_background, daemon=True)
    thread.start()

    # Return immediately with job_id for polling
    return JSONResponse({
        "job_id": job_id,
        "status": "processing",
        "message": "Generation started. Poll /jobs/{job_id} for progress.",
        "style": style,
        "mode": mode or os.environ.get("GENERATION_MODE", "fast"),
        "max_frames": max_frames,
    })


@app.get("/download/{job_id}")
def download_video(job_id: str):
    """Downloads a completed video by job ID."""
    video_path = os.path.join(TEMP_DIR, job_id, "output.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(404, "Video not found")
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"audiostory_{job_id}.mp4"
    )


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """
    Returns metadata and live progress about a job.

    If the job is still running, returns the current status and logs from
    the in-memory job registry. If the job has completed (or on server
    restart), falls back to reading files from the filesystem.
    """
    # Check in-memory job registry first (for live progress)
    job = get_job(job_id)
    if job is not None:
        job_dict = job.to_dict()
        # Merge in filesystem data (for completed jobs that may have files)
        job_dir = os.path.join(TEMP_DIR, job_id)
        if os.path.exists(job_dir):
            for fname in sorted(os.listdir(job_dir)):
                fpath = os.path.join(job_dir, fname)
                if os.path.isfile(fpath):
                    job_dict.setdefault("files", []).append({
                        "name": fname,
                        "size_bytes": os.path.getsize(fpath)
                    })
            for json_file in ["audio_data.json", "storyboard.json", "frameset.json"]:
                jpath = os.path.join(job_dir, json_file)
                if os.path.exists(jpath):
                    with open(jpath) as f:
                        job_dict[json_file.replace(".json", "")] = json.load(f)
            video_path = os.path.join(job_dir, "output.mp4")
            job_dict["video_available"] = os.path.exists(video_path)
            if job_dict["video_available"]:
                job_dict["video_size_bytes"] = os.path.getsize(video_path)
        return JSONResponse(job_dict)

    # Fallback: check filesystem only (for jobs that completed before server restart)
    job_dir = os.path.join(TEMP_DIR, job_id)
    if not os.path.exists(job_dir):
        raise HTTPException(404, "Job not found")

    result = {"job_id": job_id, "status": "completed", "files": []}

    # List all files in job directory
    for fname in sorted(os.listdir(job_dir)):
        fpath = os.path.join(job_dir, fname)
        if os.path.isfile(fpath):
            result["files"].append({
                "name": fname,
                "size_bytes": os.path.getsize(fpath)
            })

    # Load and include key JSON files
    for json_file in ["audio_data.json", "storyboard.json", "frameset.json"]:
        jpath = os.path.join(job_dir, json_file)
        if os.path.exists(jpath):
            with open(jpath) as f:
                result[json_file.replace(".json", "")] = json.load(f)

    # Check if video exists
    video_path = os.path.join(job_dir, "output.mp4")
    result["video_available"] = os.path.exists(video_path)
    if result["video_available"]:
        result["video_size_bytes"] = os.path.getsize(video_path)

    return JSONResponse(result)


# ─────────────────────────────────────────────────────────────────────────────
# Debug / Testing Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/test/stage1")
async def test_stage1(
    audio: UploadFile = File(...),
    whisper_model: str = Form("base")
):
    """Run ONLY Stage 1 (audio analysis) for debugging. Returns the analysis JSON."""
    upload_id = str(uuid.uuid4())[:8]
    upload_path = os.path.join(
        TEMP_DIR, f"test_{upload_id}{os.path.splitext(audio.filename)[1]}"
    )

    with open(upload_path, "wb") as f:
        content = await audio.read()
        f.write(content)

    try:
        audio_data = analyze_audio(
            upload_path,
            whisper_model_size=whisper_model,
            save_path=os.path.join(TEMP_DIR, f"test_{upload_id}_analysis.json")
        )
        return JSONResponse(audio_data.model_dump())
    except Exception as e:
        raise HTTPException(500, f"Stage 1 failed: {str(e)}")
    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)


@app.post("/test/stage2")
async def test_stage2(
    audio: UploadFile = File(...),
    style: str = Form("cyberpunk"),
    whisper_model: str = Form("base"),
    mode: str = Form("fast"),
    max_frames: int = Form(None)
):
    """Run Stage 1 + Stage 2 (audio analysis + storyboard) for debugging."""
    if mode not in ("fast", "quality"):
        raise HTTPException(400, "mode must be 'fast' or 'quality'")

    if style not in LORA_STYLES:
        raise HTTPException(400, f"Invalid style. Choose from: {list(LORA_STYLES.keys())}")

    upload_id = str(uuid.uuid4())[:8]
    upload_path = os.path.join(
        TEMP_DIR, f"test_{upload_id}{os.path.splitext(audio.filename)[1]}"
    )

    with open(upload_path, "wb") as f:
        content = await audio.read()
        f.write(content)

    try:
        # Stage 1
        analysis_path = os.path.join(TEMP_DIR, f"test_{upload_id}_analysis.json")
        audio_data = analyze_audio(upload_path, whisper_model_size=whisper_model,
                                   save_path=analysis_path)

        # Stage 2
        style_info = get_style(style)
        storyboard = generate_storyboard(
            audio_data=audio_data,
            style_name=style_info["display_name"],
            style_description=style_info["description"],
            style_trigger_word=style_info["trigger_word"],
            save_path=os.path.join(TEMP_DIR, f"test_{upload_id}_storyboard.json"),
            generation_mode=mode,
            max_frames=max_frames or int(os.environ.get("MAX_FRAMES", "15")) or None
        )

        return JSONResponse({
            "audio_analysis": audio_data.model_dump(),
            "storyboard": storyboard.model_dump()
        })
    except Exception as e:
        raise HTTPException(500, f"Stage 2 test failed: {str(e)}")
    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)


@app.post("/jobs/{job_id}/cancel")
def cancel_job_endpoint(job_id: str):
    """Requests cancellation of a running pipeline job. Sets a flag the pipeline polls."""
    success = cancel_job(job_id)
    if not success:
        raise HTTPException(404, "Job not found or already completed")
    return JSONResponse({"job_id": job_id, "status": "cancelling", "message": "Cancellation requested. The pipeline will stop after the current frame."})


# ─────────────────────────────────────────────────────────────────────────────
# Static file serving for frontend
# ─────────────────────────────────────────────────────────────────────────────

# Mount frontend at /app
frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_path):
    app.mount("/app", StaticFiles(directory=frontend_path, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
