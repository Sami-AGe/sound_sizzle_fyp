# AudioStory — CURRENT_STATE.md
# Last updated: v21 (2026-10-13)
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

---

## ⚠️ READ THIS BEFORE TOUCHING ANYTHING — API CALL BUDGET

**The user runs out of API calls fast. Exploration burns most of them. Do not waste calls.**

Rules:
- **Do not read a file unless you are about to edit it.**
- **Do not launch parallel agents to "explore the codebase."**
- **Do not read a whole file when you only need one function.** Use `grep` or `view` with a line range.
- **Do not re-verify things already confirmed.**
- **Do not run tests before making edits.** Make the edit, then run the minimum test.
- **For Task 2:** The codebase exploration is already complete. Do not re-explore.

---

## Open Tasks — DO IN THIS ORDER

### TASK 1 — Issue A fixes ✅ COMPLETE (committed)

Commit: `a515d1d` "Task 1 complete: guidance_scale wired, 0.22→0.45, ComfyUI integration..."

### TASK 2 — Beat-First Frame Timing Architecture ✅ COMPLETE

**Step 1 (Stage 1):** ✅ `rms_timestamps` already produced by `stage1_audio.py` (lines 141-142, 187-188). No changes needed — verified via `audio_data.json`.

**Step 2 (Schemas):** ✅ Added `hold_duration: Optional[float] = None` to `FrameDirective` and `GeneratedFrame` in `models/schemas.py`.

**Step 3 (Beat utility):** ✅ New file `stages/beat_utils.py` with `compute_beat_aligned_frame_durations(beat_times, frame_directives, total_duration)`. Snaps each frame transition to nearest beat hit; falls back to original timestamp when beats are sparse. All durations sum to ~total_duration.

**Step 4 (Stage 2):** ✅ `stage2_storyboard.py` calls utility after `parse_stored_json()` — flattens all scene frames, computes durations, sets `hold_duration` on each `FrameDirective`.

**Step 5 (Stage 4):** ✅ `stage4_video.py` uses `frame.hold_duration or frame.duration` for clip length, and accumulated `running_start` for clip positioning (replacing `clip.with_start(frame.timestamp)`). Transitions now land near beat hits.

**Status:** Beat alignment tested with real `audio_data.json` — transitions within beat range land within 250ms of beat hits (theoretical min given ~0.5s beat spacing). Durations sum to exactly `total_duration`.

### TASK 3 — Fix Download Button Clearing Video Player 🟡 (not started)

Files: `main.py`, `frontend/index.html`

### TASK 4 — Model Registry + Browser Model Selection 🟡 (not started)

### TASK 5 — Real Browser Test (not done)

---

## Key Fixed Bugs (don't re-fix these)

| Bug | Fix |
|---|---|
| Bug 6 | Frame files overwrote each other → Global `frame_progress` counter |
| Bug 7 | ComfyUI `/upload` → 405 → Changed to `/upload/image` |
| Bug 8 | Groq TPM 413 → `max_tokens=6000` |
| Bug 9 | job_id mismatch → /download 404 → `pipeline.py` uses `progress_job_id` as file-path id |
| Bug 10 | `guidance_scale=guidance_scale` NameError → Fixed to `guidance` |

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
Groq free tier: 8000 TPM. Do not raise max_tokens above 7200.
