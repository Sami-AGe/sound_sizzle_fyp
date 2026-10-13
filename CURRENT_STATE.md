# AUDIOSTORY — CURRENT_STATE.md
# v22 (2026-10-13). Rewritten to match actual git history.
# READ THIS FIRST. Only read HANDOFF.md for *historical* context.

---

## What This Project Is
30-second audio clip → AI beat-synced video. Four-stage pipeline:
Stage 1 (audio analysis, librosa/Whisper) → Stage 2 (storyboard via Groq LLM) →
Stage 3 (images via ComfyUI locally) → Stage 4 (video assembly via MoviePy)

---

## ⚠️ STATUS SNAPSHOT — git log beats stale docs

The previous `CURRENT_STATE.md` was stale on **Tasks 2 and 3**. Always cross-check
`git log` before doing work. Completed-but-apparently-"pending" tasks were already
in committed code (Task 2: `ce09a68`) or in the working tree (Task 3, committed by
you/`me` here as `1e996b1`).

| Task | Status | Commit / File |
|---|---|---|
| Task 1 — guidance_scale, 0.22→0.45, ComfyUI integration, short_anchor_text, Bug 6-10 | ✅ DONE | `a515d1d` (original) + `1e996b1` (clean working tree) |
| Task 2 — beat-first frame timing | ✅ DONE | `ce09a68` (`stages/beat_utils.py`) + docs `6ba9b25` |
| Task 3 — download button clears video player | ✅ DONE | `1e996b1` — `main.py:201` `/stream`, `frontend:1658/1680` anchor-download |
| Task 4 — model registry + browser model selection | ⬜ **START HERE** | `models/comfyui_registry.py` (new), `main.py`, `pipeline.py`, `stages/stage3_images.py`, `comfyui_client.py`, `frontend/index.html` |
| Task 5 — real browser test | ⬜ NOT DONE | needs ComfyUI (`10.40.12.55:8188`) reachable |

**Task 2 implementation (already done — do NOT redo):** `stages/beat_utils.py` holds
`compute_beat_aligned_frame_durations(beat_times, frame_directives, total_duration,
min_hold=0.5, max_hold=4.0)`. Wired: stage2 sets `FrameDirective.hold_duration`,
stage3 forwards to `GeneratedFrame`, stage4 uses `frame.hold_duration or frame.duration`
(line 77) with a `running_start` accumulator; also reads rms/onset at each frame ts.

**Task 3 implementation (already done — do NOT redo):** `GET /stream/{job_id}` returns
the mp4 via `FileResponse` *without* `filename=` ⇒ no `Content-Disposition: attachment`
⇒ inline playback survives. Frontend sets `v.src = /stream/{id}` and the download
button uses an anchor-click to force download without navigation.

---

## Paths & Stack

| Thing | Value |
|---|---|
| Project root | `/Users/syedsunny/sound_sizzle_fyp/` |
| Backend cwd | `backend/` |
| Venv python | `/Users/syedsunny/.venv/bin/python3` |
| Temp/output dir | `backend/temp/{job_id}/` |
| ComfyUI | `http://10.40.12.55:8188` (Windows PC via ZeroTier, needs `--listen`) |
| ComfyUI model | `meinamix_v12Final.safetensors` |
| Start server | `cd backend && FRAME_DELAY=2.0 .venv/bin/python3 -m uvicorn main:app --reload --port 8000` |
| Run sim test | `cd backend && python3 -u browser_sim_test.py 2>&1 \| tee /tmp/sim_output.txt` |

---

## What Is Verified Working
- All 4 stages pass individually via test scripts.
- Task 1 fully committed; Task 2 beat_utils wiring complete & verified.
- Task 3 `/stream` + anchor-download verified in source.
- Stage 3 Fast: ComfyUI text_to_image (6/6 frames, unique files).
- Stage 3 Quality: ComfyUI img2img (reference + frames).
- Full API pipeline: job submits → polls → `/download` returns MP4.
- Async pipeline with polling, cancellation, progress tracking.
- Frontend: progress bar, audio slicer, video player, ComfyUI settings panel.

---

## ⚠️ API CALL BUDGET
Do not waste calls. `git log`/`git show` are cheaper than re-reading code.

Rules:
- Don't read a file unless editing it. Trust this file + HANDOFF.md.
- Don't launch parallel "explore the codebase" agents.
- Don't read a whole file when you only need one function (use grep / line ranges).
- Don't re-verify committed work. Task 2 & 3 are committed — do NOT redo them.
- Make the edit, then run the *one* smallest test that proves it.

---

## Open Tasks — DO IN THIS ORDER

### 🟡 TASK 4 — Model Registry + Browser Model Selection  ← START HERE

`COMFYUI_MODEL` hardcoded in `.env`. Build:

1. **`models/comfyui_registry.py`** — per-checkpoint entries with: display_name, filename,
   optimal steps / guidance / denoise / width / height / sampler / scheduler, notes.
2. **`main.py`** — add `GET /models` endpoint returning the registry.
3. **`frontend/index.html`** — model cards that auto-fill ComfyUI settings.
4. **New frontend controls**: `sampler`, `scheduler`, `denoise_continuation`
   (currently only `cfDenoise` exists; no sampler/scheduler/model select).
5. Flow params through `pipeline.py` → `stages/stage3_images.py` → `comfyui_client.py`.

Files: `models/comfyui_registry.py` (new), `main.py`, `pipeline.py`,
`stages/stage3_images.py`, `comfyui_client.py`, `frontend/index.html`

### ⬜ TASK 5 — Real Browser Test
Once Task 4 done: open `http://localhost:8000/app`, submit a job via UI, verify
progress bar / video player / download. Requires ComfyUI (`10.40.12.55:8188`) reachable.

---

## Key Fixed Bugs (don't re-fix)
| Bug | What it was | Fix |
|---|---|---|
| Bug 6 | Frame files overwrote each other (per-scene frame_id) | Global `frame_progress` counter |
| Bug 7 | ComfyUI `/upload` → 405 | `/upload/image` with correct params |
| Bug 8 | Groq TPM 413 (max_tokens=8192 > 8000) | `max_tokens=6000` (stage2) |
| Bug 9 | job_id mismatch → /download 404 | `pipeline.py` uses `progress_job_id` as file-path id |
| Bug 10 | `guidance_scale=guidance_scale` NameError (ref img) | `guidance_scale=guidance` |

---

## Key Files
```
backend/
  main.py              — FastAPI app, all endpoints (/stream, /download, /jobs, /models)
  pipeline.py          — orchestrates all 4 stages
  job_manager.py       — in-memory job tracking, progress logs
  comfyui_client.py    — ComfyUI HTTP client (workflow builder)
  stages/
    stage1_audio.py    — librosa + Whisper (rms_values, rms_timestamps, onset_times)
    stage2_storyboard.py— Groq LLM storyboard, calls beat_utils
    stage3_images.py   — ComfyUI/HF generation (Fast + Quality), holds guidance/steps/etc.
    stage4_video.py    — MoviePy assembly, consumes hold_duration
    beat_utils.py      — beat-aligned frame timing (DONE)
  models/
    schemas.py         — hold_duration on FrameDirective + GeneratedFrame
    lora_registry.py   — visual style definitions
  browser_sim_test.py  — full API simulation test
  test_stage3_fast.py  — Fast mode Stage 3
  test_stage3_quality.py— Quality mode Stage 3
  recover_job.py       — one-off: regenerate missing frames + run Stage 4
frontend/
  index.html           — progress bar, audio slicer, video player, ComfyUI settings panel
```

---

## How to Update This File
Rewrite whenever git state changes. Cross-check `git log --oneline` before declaring a
task "done" — working-tree edits that look "uncommitted" may already be implemented.
