# AudioStory — CURRENT_STATE.md
# Last updated: v20 (2026-10-13)
# READ THIS FIRST. Only read HANDOFF.md if you need historical context on a specific decision.

---

## What This Project Is

30-second audio clip → AI beat-synced video. Four-stage pipeline:
Stage 1 (audio analysis, librosa/Whisper) → Stage 2 (storyboard via Groq LLM) →
Stage 3 (images via ComfyUI locally) → Stage 4 (video assembly via MoviePy)

---

## Paths & Stack

| Thing | Value |
|---|---|
| Project root | `/Users/syedsunny/sound_sizzle_fyp/` |
| Backend | `backend/` |
| Frontend | `frontend/index.html` |
| Venv python | `/Users/syedsunny/sound_sizzle_fyp/.venv/bin/python3` |
| Temp/output dir | `backend/temp/` |
| ComfyUI | `http://10.40.12.55:8188` (Windows PC via ZeroTier, needs `--listen` flag) |
| ComfyUI model | `meinamix_v12Final.safetensors` |
| Start server | `cd backend && FRAME_DELAY=2.0 .venv/bin/python3 -m uvicorn main:app --reload --port 8000` |
| Run test | `cd backend && python3 -u browser_sim_test.py 2>&1 \| tee /tmp/sim_output.txt` |

---

## What Is Verified Working

- All 4 stages pass individually via test scripts
- Fast mode Stage 3: ComfyUI text_to_image confirmed (6/6 frames, unique files)
- Quality mode Stage 3: ComfyUI img2img confirmed (reference + frames, non-trivial sizes)
- Full API pipeline: job submits → polls → /download returns MP4
- Async pipeline with job polling, cancellation, progress tracking
- Frontend: built with progress bar, audio slicer, video player, ComfyUI settings panel
- `browser_sim_test.py` replicates full frontend flow via curl
- Task 1 edits confirmed complete via `git diff` + `grep` — all changes in working tree

---

## ⚠️ READ THIS BEFORE TOUCHING ANYTHING — API CALL BUDGET

**The user runs out of API calls fast. Exploration burns most of them. Do not waste calls.**

Rules:
- **Do not read a file unless you are about to edit it.** This file and HANDOFF.md contain
  enough context to act. Trust them.
- **Do not launch parallel agents to "explore the codebase."** Previous sessions burned
  entire budgets this way without writing a single line of committed code.
- **Do not read a whole file when you only need one function.** Use `grep` or `view` with
  a line range to jump directly to the relevant section.
- **Do not re-verify things already confirmed.** If this file says something is ✅ done,
  it is done. Do not re-read the file to check.
- **Do not run tests before making edits.** Make the edit, then run the minimum test
  that proves it works. One test, not three.
- **For Task 2:** The codebase exploration is already complete (see Task 2 section below).
  Do not re-read stage1_audio.py, schemas.py, stage2_storyboard.py, or stage4_video.py
  "to understand the architecture." The plan is already written out step by step —
  open the first file you need to edit and start writing code.

The correct pattern is: read this file → commit uncommitted changes → edit → test → done. 

---

## ⚠️ COMMIT BEFORE ANYTHING ELSE

All changes from Tasks completed so far are **uncommitted**. First thing to do:

```bash
cd /Users/syedsunny/sound_sizzle_fyp
git add -A
git commit -m "Task 1 complete: guidance_scale wired, 0.22→0.45, ComfyUI integration, short_anchor_text, Bug 8+9+10 fixes"
```

---

## Open Tasks — DO IN THIS ORDER

### TASK 1 — Issue A fixes in `backend/stages/stage3_images.py` ✅ COMPLETE

Confirmed via `git diff` and `grep` in the most recent session. All changes are in the
working tree (not yet committed — see above). Do NOT re-apply any of these:

- `guidance_scale=guidance` on reference image `text_to_image()` call (line ~299) ✅
- `guidance_scale=guidance` on keyframe `image_to_image()` call (line ~357) ✅
- `guidance_scale=guidance` on continuation `image_to_image()` call (line ~381) ✅
- `guidance_scale=guidance` on fallback `text_to_image()` call (line ~394) ✅
- Both `0.22` → `0.45` in `generate_all_frames_quality()` ✅
- `FRAME_DELAY` constant added; `time.sleep()` calls updated ✅
- `resolve_prompt_tokens()` uses `short_anchor_text` not `full_anchor_text` ✅
- `schemas.py`: `short_anchor_text` field added to `CharacterAnchor` ✅
- `stage2_storyboard.py`: LLM prompt + required_keys + fallback for `short_anchor_text` ✅
- `stage2_storyboard.py`: `max_tokens` reduced to 6000 (Bug 8 fix) ✅
- `pipeline.py`: `progress_job_id` used as file-path id (Bug 9 fix) ✅
- `comfyui_client.py`: `/upload` → `/upload/image` with correct params (Bug 7 fix) ✅
- New: `browser_sim_test.py` (full API simulation), `recover_job.py` (recovery tool) ✅

The `grep` output confirmed: no `0.22` remains; `guidance_scale=guidance` appears at
lines 200, 299, 357, 381, 394 — all correct.

---

### TASK 2 — Beat-First Frame Timing Architecture 🔴 START HERE

**Exploration is already done** (three parallel agents ran last session and completed).
Do NOT re-explore. Go straight to implementation.

**What the last session established before dropping:**
- The model was in plan mode having explored `stage1_audio.py`, `schemas.py`,
  `stage2_storyboard.py`, and `stage4_video.py`
- It was mid-check on whether `rms_timestamps` is already a field on
  `AudioAnalysisResult` in `temp/85258bf0/audio_data.json` — **check this first**
  before implementing Stage 1 changes (it may already be there)

**The plan (implement in this order):**

1. **`stage1_audio.py`** — add RMS energy curve (100ms windows) to `AudioAnalysisResult`
   if not already present. Check `audio_data.json` first:
   ```bash
   cd /Users/syedsunny/sound_sizzle_fyp/backend
   python3 -c "import json; d=json.load(open('temp/85258bf0/audio_data.json')); \
     print('rms_timestamps:', 'rms_timestamps' in d); print('keys:', list(d.keys()))"
   ```

2. **`models/schemas.py`** — add `hold_duration: float` to `FrameDirective` so each
   frame carries its beat-aligned display duration.

3. **New utility** (add to `stage2_storyboard.py` or a new `beat_utils.py`):
   ```python
   def compute_beat_aligned_frame_durations(beat_times, n_frames, total_duration):
       # Snaps each frame's start/end to the nearest beat hit
       # Returns list of hold durations that sum to ~total_duration
   ```

4. **`stage2_storyboard.py`** — call the utility after storyboard generation;
   populate `FrameDirective.hold_duration` for every frame.

5. **`stage4_video.py`** — consume `hold_duration` from each `GeneratedFrame`
   instead of computing durations independently; fire effects at exact beat
   timestamps, not at frame boundaries.

**Goal:** every frame transition lands within 33ms of a real beat hit.

**Files:** `stage1_audio.py`, `models/schemas.py`, `stage2_storyboard.py`,
`stage4_video.py`, possibly new `beat_utils.py`

---

### TASK 3 — Fix Download Button Clearing Video Player 🟡 (not started)

`window.open(/download/{id})` triggers Content-Disposition:attachment — some browsers
treat this as navigation, resetting the video element's buffer.

Fix: Add `/stream/{job_id}` endpoint to `main.py` (no Content-Disposition header).
Set video player src to `/stream/`. Change download button to anchor-click:
```javascript
const a = document.createElement('a');
a.href = `${API_BASE}/download/${result.job_id}`;
a.download = `audiostory_${result.job_id}.mp4`;
document.body.appendChild(a); a.click(); document.body.removeChild(a);
```
Files: `main.py`, `frontend/index.html`

---

### TASK 4 — Model Registry + Browser Model Selection 🟡 (not started)

Model hardcoded in `.env`. Create `backend/models/comfyui_registry.py` with entries
per checkpoint (display_name, filename, optimal steps/guidance/denoise/width/height/
sampler/scheduler, notes). Add `GET /models` endpoint. Frontend: model cards that
auto-fill ComfyUI settings. Also expose `sampler`, `scheduler`, `denoise_continuation`
as new frontend controls.
Files: `models/comfyui_registry.py` (new), `main.py`, `pipeline.py`,
`stages/stage3_images.py`, `comfyui_client.py`, `frontend/index.html`

---

### TASK 5 — Real Browser Test (not done)

Once Tasks 1–3 are done: open `http://localhost:8000/app`, submit a job through the UI,
verify progress bar, video player, and download button all work correctly.

---

## Key Fixed Bugs (don't re-fix these)

| Bug | What it was | Fix |
|---|---|---|
| Bug 6 | Frame files overwrote each other (per-scene frame_id) | Global `frame_progress` counter |
| Bug 7 | ComfyUI `/upload` → 405 | Changed to `/upload/image` with correct params |
| Bug 8 | Groq TPM 413 (max_tokens=8192 > 8000 limit) | Reduced to `max_tokens=6000` |
| Bug 9 | job_id mismatch → /download 404 | `pipeline.py`: use `progress_job_id` as file-path id |
| Bug 10 | `guidance_scale=guidance_scale` NameError in reference img call | Fixed to `guidance_scale=guidance` |

---

## Environment

```
USE_COMFYUI=true
COMFYUI_HOST=http://10.40.12.55:8188
COMFYUI_MODEL=meinamix_v12Final.safetensors
GENERATION_MODE=fast
MAX_FRAMES=15
FRAME_DELAY=2.0
WHISPER_MODEL_SIZE=base
```
Groq free tier: 8000 TPM limit. Do not raise max_tokens above 7200.

---

## Key Files

```
backend/
  main.py                          — FastAPI app, all endpoints
  pipeline.py                      — orchestrates all 4 stages
  job_manager.py                   — in-memory job tracking, progress logs
  comfyui_client.py                — ComfyUI HTTP client (workflow builder)
  stages/
    stage1_audio.py                — librosa + Whisper audio analysis
    stage2_storyboard.py           — Groq LLM storyboard + character anchor
    stage3_images.py               — ComfyUI/HF image generation (Fast + Quality)
    stage4_video.py                — MoviePy video assembly + beat effects
  models/
    schemas.py                     — Pydantic models (CharacterAnchor, Storyboard, etc.)
    lora_registry.py               — visual style definitions
  browser_sim_test.py              — full API simulation test (run with python3 -u)
  test_stage3_fast.py              — Fast mode Stage 3 test
  test_stage3_quality.py           — Quality mode Stage 3 test
  recover_job.py                   — one-off: regenerate missing frames + run Stage 4
frontend/
  index.html                       — full UI (audio slicer, progress, video player)
```

---

## How To Update This File

When your session ends, rewrite this file to reflect current state.
Keep it under 120 lines. Move anything that needs preserving to HANDOFF.md first.
The prompt for the next model should be:
"Read CURRENT_STATE.md first. Only read HANDOFF.md if you need historical context."