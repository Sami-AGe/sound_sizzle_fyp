# AudioStory

> Transform 30-second audio clips into beat-synced AI-generated video stories.

AudioStory takes a short audio clip, analyzes its emotional and rhythmic structure, generates an AI-written storyboard, produces stylized images with consistent characters, and syncs visual effects to the audio's beats and drops.

## Quick Start

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Configure API keys
cp .env.example .env  # then edit .env with your keys

# 3. Start the backend API (ComfyUI must be running on http://10.40.12.55:8188)
cd backend && FRAME_DELAY=8.0 /Users/syedsunny/sound_sizzle_fyp/.venv/bin/python3 -m uvicorn main:app --reload --port 8001

# 4. Open the frontend
# Visit http://localhost:8001/app/ for the UI
# Or use the API directly at http://localhost:8001/docs
```

---

## How It Works

```
Audio Clip (≤30s)
    ↓ Stage 1: Audio Analysis (librosa + Whisper)
    Tempo, beats, onsets, RMS energy, spectral centroid, mood, transcription, unique moments
    ↓ Stage 2: Storyboard Generation (Gemini 3.5-flash-lite, dual-call)
    Call 1: Locked character anchor (hair, eyes, clothing, body, distinguishing, short_anchor_text)
    Call 2: Full scene/frame plan (6–8 scenes, 3–5 frames each, keyframe + continuations)
    ↓ Stage 3: Image Generation (ComfyUI local — primary)
    Fast mode:   ALL frames → text-to-image (SD3-Medium, seed-locked per scene)
    Quality mode: Reference image + img2img (FLUX.1-Kontext, pixel inheritance)
    ↓ Stage 4: Video Assembly (OpenCV + MoviePy v2)
    6 beat-synced effects + mood color grade + storyboard-driven scene transitions
    → 1080×1080 MP4 video output
```

---

## Architecture

| Stage | Component | Stack |
|-------|-----------|-------|
| 1 | Audio Analysis | librosa + Whisper (Groq API) |
| 2 | Storyboard | **Gemini 3.5-flash-lite** (native JSON mode, 1500 req/day free tier) |
| 3 | Image Generation | **ComfyUI** (local, Windows via ZeroTier) — SD3-Medium (txt2img) + FLUX.1-Kontext (img2img) |
| 4 | Video Assembly | OpenCV (effects) + MoviePy v2 (composition) |
| Backend API | FastAPI | Python 3.9+ |
| Frontend | HTML/CSS/JS | Vanilla JS (no framework), served at `/app/` |

### Dual-Mode Generation

| Mode | Method | Consistency | Best For |
|------|--------|-------------|----------|
| **Fast** | Every frame = text-to-image (SD3-Medium) | Character Anchor tokens + scene seed | Speed, lower VRAM, consistent style |
| **Quality** | Reference image + img2img (FLUX.1-Kontext) | Pixel inheritance from previous frame | Maximum visual fidelity, smooth motion |

---

## Visual Styles

| Style Key | Display Name | Trigger Word | LoRA File | Description |
|-----------|-------------|--------------|-----------|-------------|
| `mha_anime` | MHA Anime | `mha_anime_style` | `mha_anime_v1.safetensors` | Shonen anime, vibrant cel-shaded |
| `cyberpunk` | Cyberpunk Neon | `cyberpunk_style` | `cyberpunk_v2.safetensors` | Dark neon-lit, futuristic city |
| `lofi_pastel` | Lofi Pastel | `lofi_pastel_style` | `lofi_pastel_v1.safetensors` | Soft pastel, warm illustration |
| `dark_fantasy` | Dark Fantasy | `dark_fantasy_style` | `dark_fantasy_v1.safetensors` | Epic dark fantasy, dramatic lighting |
| `minimalist` | Minimalist | `minimalist_style` | `minimalist_v1.safetensors` | Clean flat design, geometric |
| `watercolor` | Watercolor | `watercolor_style` | `watercolor_v1.safetensors` | Soft watercolor washes |
| `retro_anime` | Retro Anime | `retro_anime_style` | `retro_anime_v1.safetensors` | 90s anime aesthetic, VHS grain |
| `none` | Base Model Only | — | — | No LoRA, pure base checkpoint |

Styles are defined in `backend/models/lora_registry.py` with per-style LoRA weights, trigger words, and Stage 2 direction hints.

---

## Project Structure

```
sound_sizzle_fyp/
├── backend/
│   ├── main.py                      # FastAPI entry point (/generate, /stream, /models, /styles, /health)
│   ├── pipeline.py                  # Orchestrates Stages 1–4
│   ├── comfyui_client.py            # ComfyUI HTTP client + workflow builder
│   ├── job_manager.py               # Thread-safe job registry + SSE progress
│   ├── models/
│   │   ├── schemas.py               # Pydantic data contracts (AudioAnalysisResult, Storyboard, GeneratedFrame, etc.)
│   │   ├── lora_registry.py         # Style + mood definitions, LoRA metadata
│   │   └── comfyui_registry.py      # Checkpoint registry (SD3, FLUX, Meinamix, etc.)
│   ├── stages/
│   │   ├── stage1_audio.py          # librosa analysis + Whisper transcription + unique moment detection
│   │   ├── stage2_storyboard.py     # Gemini dual-call: character anchor + full storyboard
│   │   ├── stage3_images.py         # Fast/Quality image generation via ComfyUI
│   │   ├── stage4_video.py          # Beat-synced effects + mood grade + scene transitions
│   │   └── beat_utils.py            # Beat-aligned frame durations (do not modify)
│   └── utils/
│       ├── prompt_builder.py        # Mode-aware storyboard prompts (Fast/Quality schemas)
│       └── audio_utils.py           # Energy/brightness/onset helpers
├── frontend/
│   └── index.html                   # Full UI: upload, style picker, param panel, SSE progress, video player
├── .env                             # API keys + runtime config (see below)
├── requirements.txt
├── CURRENT_STATE.md                 # Living task/bug tracker (read first)
└── README.md
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check (ComfyUI connectivity) |
| `GET` | `/styles` | List available visual styles with metadata |
| `GET` | `/models` | List available ComfyUI checkpoints |
| `POST` | `/generate` | Run full pipeline (multipart: audio, style, mode, max_frames, gen_params) |
| `GET` | `/stream?job_id=...` | SSE real-time progress for a job |
| `GET` | `/download/{job_id}` | Download generated video |
| `GET` | `/jobs/{job_id}` | Get job details, artifacts, logs |
| `POST` | `/test/stage1` | Run only audio analysis |
| `POST` | `/test/stage2` | Run audio + storyboard generation |

### Generate Request (multipart/form-data)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | file | Yes | Audio file (≤30s, mp3/wav) |
| `style` | string | Yes | Style key (e.g., `cyberpunk`) |
| `mode` | string | No | `fast` or `quality` (default: `fast`) |
| `max_frames` | int | No | Target frame count (default: 30, env `MAX_FRAMES`) |
| `gen_params` | JSON | No | Overrides: `{"steps":20,"guidance_scale":7.5,"width":768,"height":768,"denoise":0.82,"denoise_continuation":0.65,"seed":-1,"model":"meinamix","sampler":"euler","scheduler":"normal"}` |

---

## Environment Variables

```env
# ── Required API Keys ──────────────────────────────────────────────────────
GEMINI_API_KEY=your_gemini_api_key          # From https://aistudio.google.com/apikey
GROQ_API_KEY=your_groq_api_key              # For Stage 1 Whisper only
HF_TOKEN=your_huggingface_token             # Fallback if USE_COMFYUI=false

# ── ComfyUI Connection ─────────────────────────────────────────────────────
COMFYUI_URL=http://10.40.12.55:8188         # ZeroTier Windows PC with --listen
USE_COMFYUI=true                            # Set false to use HuggingFace fallback

# ── Model Selection (ComfyUI checkpoint filenames) ─────────────────────────
COMFYUI_MODEL=meinamix_v12Final.safetensors  # Default txt2img checkpoint
TEXT_TO_IMAGE_MODEL=stabilityai/stable-diffusion-3-medium   # HF model ID (ignored if ComfyUI)
IMAGE_TO_IMAGE_MODEL=black-forest-labs/FLUX.1-Kontext-dev   # HF model ID (ignored if ComfyUI)

# ── Generation Defaults ────────────────────────────────────────────────────
MAX_FRAMES=30                               # Target total frames (6–8 scenes × 3–5 frames)
GENERATION_MODE=fast                        # fast | quality
FRAME_DELAY=8.0                             # Seconds between frames (GPU thermal management)
WHISPER_MODEL_SIZE=base                     # tiny | base | small | medium
OUTPUT_DIR=./temp                           # Job output directory

# ── Optional ───────────────────────────────────────────────────────────────
MAX_AUDIO_DURATION=30                       # Hard limit on input audio
```

---

## Key Features

### Character Consistency (Task A, C, F, G)
- **Locked anchor**: Single LLM call generates hair, eyes, clothing, body, distinguishing traits
- **Short anchor (8–12 words)**: `teal bob hair, amber eyes, black trench coat, pink circuit patterns` — injected into every Fast mode prompt
- **Fast mode**: Text-to-image with `{character_anchor}` token + scene seed locking
- **Quality mode**: Reference image + img2img with pixel inheritance

### Beat-Synced Everything (Task D, beat_utils.py)
- Frame timestamps aligned to nearest beat
- Frame hold durations computed from beat intervals
- 6 visual effects triggered by audio features:
  1. **Beat Flash** — white flash on keyframes (energy-driven)
  2. **Glitch** — chromatic aberration + slice displacement (onset-driven)
  3. **Push-in Zoom** — progressive zoom before scene cuts
  4. **Color Grade** — warm/cool + saturation (spectral centroid + energy)
  5. **Vignette** — dark oval at low energy moments
  6. **Zoom Punch** — instant zoom at high-energy onsets

### Storyboard-Driven Scene Transitions (Task H)
Stage 2 declares `transition_from_previous` per scene; Stage 4 applies it:
- `glitch_cut` — chromatic aberration + slice displacement (intensity 0.9)
- `hard_cut` — instant frame replacement
- `flash_cut` — white flash synchronized to onset (0.25s fade)
- `dissolve` — linear crossfade with previous scene's last frame

### Smart Scene Pacing (Task G)
- Target: 1 keyframe every 3–5 seconds → **6–8 scenes for 30s audio**
- Each scene: 1 keyframe + 2–4 continuations (3–5 frames total)
- Continuation strengths: **0.50–0.88** (subtle 0.50–0.60, moderate 0.65–0.75, strong 0.75–0.88)
- Values below 0.50 are invisible in video — enforced in prompt and validation

### Unique Moment Keyframe Placement (v31)
- Stage 1 detects distinctive audio moments (energy deviation + spectral flux + onset alignment)
- Stage 2 snaps keyframe timestamps to nearest unique moment
- Precedence-based clustering prevents duplicate nearby keyframes

---

## Usage

### CLI (from `backend/` directory)

```bash
# Full pipeline
python -m backend.pipeline /path/to/audio.mp3 cyberpunk

# With overrides
MAX_FRAMES=30 GENERATION_MODE=quality python -m backend.pipeline audio.mp3 mha_anime

# Individual stages (for debugging)
python -m backend.stages.stage1_audio audio.mp3 base
python -m backend.stages.stage2_storyboard audio.mp3 cyberpunk --mode fast --max-frames 30
python -m backend.stages.stage3_images audio.mp3 cyberpunk
```

### API

```bash
# Generate video
curl -X POST "http://localhost:8001/generate" \
  -F "audio=@my_clip.mp3" \
  -F "style=cyberpunk" \
  -F "mode=fast" \
  -F "max_frames=30"

# Stream progress (SSE)
curl -N "http://localhost:8001/stream?job_id=abc123"

# Download result
curl -O "http://localhost:8001/download/abc123"
```

### Frontend UI

Open `http://localhost:8001/app/` — features:
- Audio upload with waveform preview
- Style gallery with trigger word preview
- Generation mode toggle (Fast/Quality)
- Advanced parameter panel (steps, guidance, denoise, seed, model, sampler)
- Real-time SSE progress log with stage/frame counters
- Video player with download button

---

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Install ComfyUI client deps (if not in requirements)
pip install websocket-client aiohttp

# Run import checks
cd backend && python -c "from pipeline import run_pipeline; print('OK')"

# Start dev server (with FRAME_DELAY for GPU thermal management)
cd backend && FRAME_DELAY=8.0 python -m uvicorn main:app --reload --port 8001

# View current task/bug status
cat CURRENT_STATE.md
```

---

## Requirements

- **Python 3.9+** (3.11+ recommended for Gemini SDK)
- **Gemini API key** (free tier: 1500 req/day on 3.5-flash-lite)
- **Groq API key** (free tier: for Whisper transcription only)
- **ComfyUI server** running on a Windows PC with GPU, accessible via ZeroTier (`--listen` flag)
  - Required checkpoints: `meinamix_v12Final.safetensors`, SD3-Medium, FLUX.1-Kontext-dev
  - Required LoRAs: one per style (see `backend/models/lora_registry.py`)
- **HuggingFace token** (optional, fallback only)
- **ffmpeg** (for MoviePy video encoding)

---

## Important Notes

| Topic | Detail |
|-------|--------|
| **Frame delay** | `FRAME_DELAY` is re-read from env **each frame** — restart server after changing `.env` |
| **Max frames** | Default 30. For 30s audio: 6–8 scenes × 3–5 frames = dynamic video, not slideshow |
| **Continuation strength** | 0.50–0.88 only. Below 0.50 = invisible. Prompt enforces this; validation warns. |
| **Character anchor** | Generated once (Call 1), frozen for entire pipeline. Short anchor = 8–12 words with exact colors. |
| **ComfyUI model param** | HuggingFace model IDs (containing `/`) are ignored; falls back to `ckpt_name` or `COMFYUI_MODEL` |
| **Gemini model** | Currently `gemini-3.5-flash-lite` (1500 req/day). Was `gemini-3.8-flash` (50 req/day). |
| **Deprecated SDK** | `google.generativeai` shows FutureWarning; migrate to `google.genai` when possible. |

---

## Task Status (from CURRENT_STATE.md)

| Task | Status |
|------|--------|
| Task 1–7 | ✅ Done |
| Task A–F | ✅ Done |
| Task G — Continuation strength + scene pacing | ✅ Done |
| Task H — Storyboard-driven scene transitions | ✅ Done (commit 9225f6c) |
| Task I — Groq → Gemini migration | ✅ Done (commit fd9ddb2) |
| **Task 6 — Real browser test** | ⬜ **Pending** |

---

## License

MIT
