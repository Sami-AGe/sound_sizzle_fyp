# AudioStory

> Transform 30-second audio clips into beat-synced AI-generated video stories.

AudioStory takes a short audio clip, analyzes its emotional and rhythmic structure, generates an AI-written storyboard, produces stylized images with consistent characters, and syncs visual effects to the audio's beats and drops.

## Quick Start

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Configure API keys
cp .env.example .env  # then edit .env with your keys

# 3. Start the backend API
cd backend && uvicorn main:app --reload

# 4. Open the frontend
# Visit http://localhost:8000/app/ for the UI
# Or use the API directly at http://localhost:8000/docs
```

## How It Works

```
Audio Clip (30s)
    ↓ Stage 1: Audio Analysis
    Tempo, beats, onsets, energy, mood, transcription
    ↓ Stage 2: Storyboard Generation (Groq LLM)
    Structured JSON: scenes, keyframes, continuations
    ↓ Stage 3: Image Generation (HuggingFace API)
    Reference image → keyframe + continuation frames
    ↓ Stage 4: Video Assembly (OpenCV + MoviePy)
    Beat-synced effects: flash, glitch, zoom, color grade, vignette
    → Beat-synced video output
```

## Architecture

| Stage | Component | Stack |
|-------|-----------|-------|
| 1 | Audio Analysis | librosa + Whisper |
| 2 | Storyboard | Groq LLM (Llama 3.3 70B) |
| 3 | Image Generation | HuggingFace Inference API (SD 2.1) |
| 4 | Video Assembly | OpenCV (effects) + MoviePy v2 (composition) |
| Backend API | FastAPI | Python 3.9+ |
| Frontend | HTML/CSS/JS | Vanilla JS (no framework) |
| Colab | Notebook | Google Colab (T4 GPU) |

## Visual Styles

| Style Key | Display Name | Description |
|-----------|-------------|-------------|
| `mha_anime` | MHA Anime | Shonen anime style, vibrant colors |
| `cyberpunk` | Cyberpunk Neon | Dark neon-lit, futuristic city |
| `lofi_pastel` | Lofi Pastel | Soft pastel, warm illustration |
| `dark_fantasy` | Dark Fantasy | Epic dark fantasy illustration |
| `minimalist` | Minimalist | Clean flat design, geometric |

## Project Structure

```
audiostory/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── pipeline.py          # Orchestrates all stages
│   ├── models/
│   │   ├── schemas.py       # Pydantic data models
│   │   └── lora_registry.py # Style definitions
│   ├── stages/
│   │   ├── stage1_audio.py     # Audio analysis
│   │   ├── stage2_storyboard.py # Groq storyboard generation
│   │   ├── stage3_images.py    # HF image generation
│   │   └── stage4_video.py     # Video effects + assembly
│   ├── utils/
│   │   ├── audio_utils.py
│   │   ├── image_utils.py
│   │   └── prompt_builder.py
│   └── loras/                # Trained LoRA files (optional)
├── frontend/
│   └── index.html
├── colab/
│   └── AudioStory_Pipeline.ipynb
├── .env                      # API keys
├── requirements.txt
└── README.md
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/styles` | List available visual styles |
| `POST` | `/generate` | Run full pipeline (audio + style → video) |
| `GET` | `/download/{job_id}` | Download generated video |
| `GET` | `/jobs/{job_id}` | Get job details and artifacts |
| `POST` | `/test/stage1` | Run only audio analysis |
| `POST` | `/test/stage2` | Run audio + storyboard generation |

## Environment Variables

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
BASE_MODEL=stabilityai/stable-diffusion-2-1
WHISPER_MODEL_SIZE=base        # tiny | base | small | medium
OUTPUT_DIR=./temp
MAX_AUDIO_DURATION=30
```

## Usage

### CLI

```bash
# Full pipeline
python -m backend.pipeline audio.mp3 mha_anime

# Individual stages
python -m backend.stages.stage1_audio audio.mp3 tiny
python -m backend.stages.stage2_storyboard audio.mp3 cyberpunk
python -m backend.stages.stage3_images audio.mp3 cyberpunk
```

### API

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "audio=@my_clip.mp3" \
  -F "style=cyberpunk"
```

### Google Colab

Open `colab/AudioStory_Pipeline.ipynb` in Google Colab for GPU-accelerated generation.

## Requirements

- Python 3.9+
- Groq API key (free tier: ~14,400 requests/day)
- HuggingFace token (free tier: rate-limited)
- Google Colab account (for GPU) or local GPU

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m backend.stages.stage1_audio temp/test_audio.wav tiny

# Start dev server
cd backend && uvicorn main:app --reload
```

## License

MIT
