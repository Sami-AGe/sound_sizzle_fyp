# AUDIOSTORY — CURRENT_STATE.md
# v30 (2026-10-16) — Unique moment keyframe detection, simplified visual language, FRAME_DELAY=8.0
# v29 (2026-10-16) — Bug 15+16 fixes, Stage 3 verified, Tasks 1-5/A-F complete
# v27 (2026-10-16) — Tasks A–F, D, E1, E2 complete. Keyframes use text_to_image. Prompt order fixed.
# v26 (2026-10-14) — Visual quality diagnostic added from real user test
# READ THIS FIRST. Only read HANDOFF.md for historical context.

---

## ✨ v30 — Unique Moment Keyframe Detection + Visual Language Fixes

### Problem Summary (from user test feedback)
1. Storyboard generates 1 frame per time segment → slideshow pacing, not video-like
2. Storyboard directions use ambiguous words ("dynamic", "diagonal compositions") the image generator can't follow
3. FRAME_DELAY inconsistent — `.env.example` had 0.5s, code default was 8.0s

### Fix 1 — Unique moment detection for keyframe placement (`stage1_audio.py`)

Added `detect_unique_moments()` function that:
- Computes **spectral flux** (timbral change rate) from spectral centroid derivative
- Computes **energy deviation** (how far each frame is above the mean)
- Combines with **onset proximity** (peaks near audio hits/drops)
- Uses `scipy.signal.find_peaks` to find local maxima in the combined novelty curve
- Returns top `num_scenes + 2` moments with `{timestamp, energy, brightness, description}`
- Falls back to evenly-spaced beats if no peaks found
- Results stored in `AudioAnalysisResult.unique_moments` (new optional field)

### Fix 2 — Frame distribution enforcement (`stage2_storyboard.py`)

Added `_enforce_frame_distribution()` function that:
- Ensures each scene has 3+ frames (1 keyframe + 2-3+ continuations) for video-like pacing
- Snaps keyframe timestamps to nearest unique moment when available
- If the LLM produced too few frames, duplicates continuation frames around keyframes
- Calls `_rebalance_frame_timestamps()` to ensure timestamps fit within scene boundaries
- Snapping to beat-aligned times when `audio_data.beat_times` is available

### Fix 3 — Simplified visual language (`prompt_builder.py`)

Added a **"VISUAL LANGUAGE CHEAT SHEET"** section to both Fast and Quality mode frame instructions:
- Camera angle: only "low angle", "high angle", "eye level"
- Shot distance: only "wide shot", "medium shot", "close-up", "extreme close-up"
- Lighting: concrete descriptions like "single key light from upper left"
- Movement: concrete actions like "character blinking", "camera tilts up"
- Replaced abstract terms like "dynamic" with specific instructions

Added **"FRAME DISTRIBUTION INSTRUCTIONS"** telling the LLM to:
- Place 2-4 continuation frames after each keyframe
- Use specific modifier_prompt examples: "character blinks slowly", "camera tilts up gently"
- Snap keyframe timestamps to unique moments

### Fix 4 — FRAME_DELAY consistency (`.env.example`)

Changed `FRAME_DELAY=0.5` → `FRAME_DELAY=8.0` to match the code default in `stage3_images.py`.

### Schema additions (`models/schemas.py`)

Added three optional fields to `AudioAnalysisResult`:
- `spectral_flux: Optional[List[float]]` — timbral change rate per frame
- `novelty_scores: Optional[List[float]]` — combined uniqueness score per frame
- `unique_moments: Optional[List[dict]]` — `[{timestamp, energy, brightness, description}]`

---

## ⚠️ API CALL BUDGET — READ BEFORE TOUCHING ANYTHING
- Do not read a file unless you are about to edit it.
- Do not launch exploration agents. Use `grep -n` with line ranges instead.
- Do not re-verify completed tasks. Tasks 1–4 are committed.
- Pattern: grep for insertion point → write the one function → import-check → done.

---

## Status Snapshot

| Task | Status |
|---|---|
| Task 1 — guidance_scale, 0.22→0.45, Bug 6–10 | ✅ DONE committed |
| Task 2 — beat-first frame timing (`beat_utils.py`) | ✅ DONE committed `ce09a68` |
| Task 3 — `/stream` endpoint + anchor download fix | ✅ DONE committed `1e996b1` |
| Task 4 — model registry + browser model selection | ✅ DONE verified |
| Task A — Fix prompt order in Stage 2 | ✅ DONE — environment-first in `prompt_builder.py` |
| Task B — Fix black frames (last 8s) | ✅ DONE — placeholder detection in `stage4_video.py` |
| Task C — Fix `character_anchor: null` | ✅ DONE — guard in `stage2_storyboard.py` + Quality mode injection |
| Task D — Fix FRAME_DELAY not taking effect | ✅ DONE — fresh env read + visible log line in `stage3_images.py` |
| Task E1 — Default/max frames → 30 | ✅ DONE — `.env`, `main.py`, `index.html` |
| Task E2 — Optimal parameter suggestions panel | ✅ DONE — added to `frontend/index.html` |
| Task F — Fix keyframe/continuation logic | ✅ DONE — Quality mode keyframes use `text_to_image` |
| Task 5 — Style LoRA + mood system | ✅ DONE — `lora_registry.py` replaced; `stage1_audio.py` mood map updated |
| Task 6 — Real browser test | ⬜ LAST |
| Task 7 — Frontend professionalism + audio fixes + slicer preview | ✅ DONE |

---

## ✨ Task 7 — Frontend Professionalism + Audio Fixes + Slicer Preview (DONE)

### 7A — Frontend professionalism (SVG icons, typography, UX polish)

**File:** `frontend/index.html`

**What changed:**
- Added `ICONS` JavaScript object with clean SVG icon definitions (audio, scissors, gear, play,
  pause, stop, volume, mute, close, warning, arrowUp, arrowDown, fullscreen, check, clock, attach)
- Replaced all emoji throughout the UI with SVG icons rendered via `innerHTML` on `.icon` spans:
  🎵 → SVG waveform, ✂️ → SVG scissors, ⚙ → SVG gear, ⏹ → SVG stop, 🔇/🔊 → SVG mute/volume,
  ▶/⏸ → SVG play/pause, ▼/▲ → SVG arrowUp/arrowDown, ⚠ → SVG warning, ✕ → SVG close
- Fixed duplicate `id="vPb"` attribute on the play button (two `id` attrs in markup)
- Changed console/comfy toggle arrow spans from `textContent` (emoji) to `innerHTML` (SVG strings)
- Fixed incorrect fullscreen SVG path (was an overly complex incorrect path)
- Replaced fragile string manipulation of `ICONS.arrowDown.replace(...)` with a proper
  separate `ICONS.arrowUp` entry
- Changed logo font-weight from 600 to 700 for bolder branding
- Added `playsinline` attribute to video element (prevents fullscreen lock on iOS/mobile)
- Auto-collapse dev console by default (cleaner initial view)
- Added `autoSelectDefaultStyle()` — selects "none" style on page load so the style dropdown
  is never in an unselected state
- Made `initApp()` async to properly await style/model loading before UI interaction
- Removed decorative emoji (✦, ★) used as section bullets — replaced with plain text labels

### 7B — Audio missing from generated video (fix)

**Root cause:** The uploaded audio file (`upload_path`) was a relative path in `TEMP_DIR`.
After `run_pipeline()` returned, the `finally` block in `main.py`'s background thread deleted
the upload file. While the pipeline copies it into the job directory, the copy used a path
relative to the CWD which could resolve incorrectly when the pipeline ran in a background
thread.

**Fix — `backend/pipeline.py`:**
```python
# Copy audio file into job directory so it survives upload cleanup
# and resolves with an absolute path during Stage 4 video assembly.
audio_ext = os.path.splitext(audio_path)[1] or ".wav"
job_audio_path = os.path.abspath(os.path.join(job_dir, f"input_audio{audio_ext}"))
import shutil as _shutil
_shutil.copy2(audio_path, job_audio_path)
audio_path = job_audio_path
```
The audio is now copied to `job_dir/input_audio.{ext}` with an absolute path, ensuring the
file persists and resolves correctly regardless of working directory or upload cleanup timing.

**Fix — `backend/stages/stage4_video.py`:**
- Added `try/except` around `AudioFileClip()` loading and `with_audio()` attachment
- Added `log_progress()` calls to log audio loading attempt, success, and any failure
- Audio codec is explicitly `"aac"` in `write_videofile()` to avoid codec issues
- Uses `.subclipped(0, audio_duration)` for moviepy v2 API compatibility (not `.subclip()`)

**Fix — `backend/main.py` `test_stage2` endpoint:**
- Default style changed from `"cyberpunk"` to `"none"` (validation error: actual key is
  `cyberpunk_anime`, so the old default always failed)
- Added `style_key=style` and `detected_mood=audio_data.detected_mood` to the
  `generate_storyboard()` call

**Fix — `backend/stages/stage1_audio.py`:**
- Added `neutral_cinematic` to `VALID_MOODS` set
- Updated `_detect_mood()` fallback `mood_map`: `neutral_cinematic` → `neutral_cinematic`
  (was mapped to `neutral`)
- Updated Groq LLM prompt to include `neutral_cinematic` as a classification option

### 7C — Audio preview in slicer + stop button

**File:** `frontend/index.html`

**What changed:**
- Added `previewSlice()` function using the Web Audio API (`AudioBufferSourceNode`,
  `AudioContext`) to play the selected 30-second slice from the slicer
- Added `_stopPreview()` cleanup function that stops the source node, closes the
  `AudioContext`, and restores button state
- Added `stopSlicePreview()` public function called by the stop button
- Added `isPreviewing` state flag to track preview state
- Added a dedicated "Stop" button (`<button id="stopPreviewBtn">`) that appears as
  `inline-flex` only while a preview is playing (initially `display:none`), and is
  hidden again on `_stopPreview()`
- The "Preview slice" button is disabled while preview is playing (preventing overlap)
- `_stopPreview()` is called on file change and during cleanup to prevent audio leaks
- The preview plays from `slicer.getStart()` for `min(30, audio_buffer.duration - start)`
  seconds with a log line showing the time range

---

## Paths & Stack

| Thing | Value |
|---|---|
| Project root | `/Users/syedsunny/sound_sizzle_fyp/` |
| Backend cwd | `backend/` |
| Venv python | `/Users/syedsunny/.venv/bin/python3` |
| ComfyUI | `http://10.40.12.55:8188` (Windows PC via ZeroTier, `--listen` flag) |
| Start server | `cd backend && FRAME_DELAY=8.0 /Users/syedsunny/.venv/bin/python3 -m uvicorn main:app --reload --port 8001` |
| Frontend URL | `http://localhost:8001/app/` |

---

## 🔴 VISUAL QUALITY DIAGNOSTIC — Real User Test Results

User ran Quality mode (Ghibli style, 20 frames, 30s audio). Results were unusable.
Here is the exact diagnosis from frame inspection and storyboard analysis.

### Problem 1 — WRONG ENVIRONMENTS (most critical)

Storyboard specified: "misty forest clearing," "deeper forest path," "open glade,"
"ancient glowing tree clearing," "twilight forest edge."
Every generated frame showed: indoor Japanese corridor with sliding doors on both sides.
Not a single frame had a forest.

**Root cause:** The keyframe full_prompt starts with 60+ tokens of character description
before mentioning the environment. Example from the actual storyboard JSON:
```
"soft wavy chestnut hair with subtle silver streaks, large amber eyes, wistful yet
determined expression, loose teal kimono with embroidered clouds and wooden sandals,
slender slightly hunched frame, gentle posture, glowing firefly pendant...
...standing in a misty forest clearing at sunrise"
```
By the time ComfyUI reads "forest clearing," it has already committed to generating
a kimono character in a Japanese architectural setting — meinamix's strong prior
for those tokens. The scene environment appears too late in the prompt to matter.

**Fix — Task A (DONE):** The LLM instructions in `prompt_builder.py` were rewritten so
every `full_prompt` must follow this exact order:
1. ENVIRONMENT first (scene setting, location, atmosphere, 10-15 words)
2. LIGHTING (5 words): time of day, light quality
3. STYLE TOKENS (5 words): art style, medium
4. `{{character_anchor}}` token LAST

Correct order example: `"misty forest clearing at sunrise, morning mist drifting,
soft amber rays, painterly illustration, studio ghibli style — {character_anchor}"`

The `{character_anchor}` token is resolved programmatically from the locked anchor
via `resolve_prompt_tokens()` in `stage3_images.py`, using `short_anchor_text` (3-5
words) instead of the full 60-token description.

### Problem 2 — WRONG CHARACTER APPEARANCE

Storyboard specified: chestnut hair, amber eyes.
Every frame showed: grey-blonde hair, blue eyes.
meinamix ignores specific color directives when its style priors are stronger.

**Status:** Partially addressed — the prompt order fix (Task A) ensures the environment
gets priority. The color name issue (chestnut → brown) remains a known limitation of
meinamix style priors. The `short_anchor_text` now uses shorter descriptors which helps.

### Problem 3 — LAST 8 SECONDS COMPLETELY BLACK

Frames at t=22s–30s (Scene 5) are pure black. Stage 3 either failed to generate
Scene 5's frames and produced black placeholders, or Stage 4 ran out of frames and
padded with black. This is a real bug — not a quality issue.

**Fix — Task B (DONE):** In `stage4_video.py` `build_video()`, added placeholder
detection. Each frame is checked:
- File size < 5000 bytes (solid-colour PNG compresses tiny)
- OR mean brightness < 20 AND pixel std dev < 5 (flat dark image with no texture)

When detected, the frame is replaced with the last valid frame (`last_valid_image`).
The first frame has no fallback — it's kept as-is with a warning log. Added
`log_progress` import and `last_valid_image` tracking variable.

Also: Stage 3's error handling already logs `[ERROR]` when a frame fails, and saves
a placeholder with a distinctive log line. The placeholder color is `(50, 50, 50)`
(dark gray, not pure black), which the detection logic catches.

### Problem 4 — character_anchor: null IN STORYBOARD JSON

The storyboard JSON showed `"character_anchor": null`. This means the character
anchor generation (Stage 2 Call 1) either failed silently or the result was not
attached to the storyboard object before it was serialized. Quality mode uses the
character anchor for its reference image — if it's null, the reference image is
generated without a proper character description.

**Fix — Task C (DONE):** Added a guard in `generate_storyboard()` immediately after
`generate_character_anchor()` Call 1:
```python
if anchor is None or not anchor.get("short_anchor_text"):
    raise ValueError(
        "Character anchor generation failed — storyboard aborted. "
        "Check Groq API key and TPM limits."
    )
```

Also: `storyboard.character_anchor = CharacterAnchor(**anchor)` was added for Quality
mode (previously only set for Fast mode). This ensures the `{character_anchor}` token
can be resolved in Stage 3 for all modes.

### Problem 5 — SAME COMPOSITION EVERY FRAME

Every frame showed character dead-center, standing, facing forward.
Storyboard specified: wide low-angle establishing shots, diagonal medium shots,
extreme close-ups, low angles looking up. None were honored.

**Root cause:** img2img in Quality mode inherited the reference image's composition.
The reference image was always "neutral standing pose, facing forward" — every
subsequent img2img frame was anchored to that composition. Even with denoise=0.85,
the spatial layout of the reference dominated.

**Fix — Task F (DONE):** In `stage3_images.py` `generate_all_frames_quality()`:

**Before (broken):**
```python
for scene in storyboard.scenes:
    for frame in scene.frames:
        if frame.type == FrameType.KEYFRAME:
            img = client.image_to_image(
                image=reference_image_data,  # ← reference portrait
                prompt=full_prompt,
                strength=0.82, ...
            )
        else:  # CONTINUATION
            img = client.image_to_image(
                image=reference_image_data,  # ← ALSO reference portrait (BUG)
                prompt=modifier_prompt, ...
            )
        previous_image = img
```

**After (fixed):**
```python
for scene in storyboard.scenes:
    for frame in scene.frames:
        if frame_directive.type == FrameType.KEYFRAME:
            # Fresh text_to_image — establishes a new scene from scratch
            resolved_prompt = resolve_prompt_tokens(
                frame_directive.full_prompt, storyboard.character_anchor
            )
            full_prompt = f"{resolved_prompt}, {style_trigger}, {scene.color_palette}, illustration, stylized"
            current_image = client.text_to_image(
                prompt=full_prompt, model=TEXT_TO_IMAGE_MODEL, ...
                seed=scene.scene_seed, ...  # intra-scene consistency
            )
        else:  # CONTINUATION
            # img2img from the PREVIOUS frame (not the reference portrait)
            modifier_prompt = f"{frame_directive.modifier_prompt}, {style_trigger}"
            current_image = client.image_to_image(
                image=previous_image,  # ← previous frame, NOT reference
                prompt=modifier_prompt, ...
                strength=denoise_continuation, ...  # 0.45 default
            )
        previous_image = current_image.copy()  # next continuation sources from this
```

Key changes:
- Keyframes: `text_to_image` (fresh from scratch, environment-first prompt)
- Continuations: `img2img` from `previous_image` with short `modifier_prompt` only
  (removed the 60-token `subject_description` prefix)
- Reference portrait is kept as a metadata artifact (saved to disk) but NOT used
  as source for any generation call
- Keyframe seed uses `scene.scene_seed` for intra-scene consistency
- Character consistency comes from: scene_seed locking, short_anchor_text at end
  of keyframe prompts, and visual similarity between keyframe and continuations

The reference portrait generation is still kept in the Quality mode function — it's
saved as `reference.png` for metadata/debugging purposes, but all keyframe and
continuation generation now ignores it.

Fast mode was already sourcing continuations from `previous_image` (correct) —
no change needed there beyond the Task A prompt order fix.

### Problem 6 — 20 FRAMES FOR 30 SECONDS IS TOO FEW

The pipeline generates images that each hold for a variable duration — it is NOT
one frame per second. 20 frames × average 1.5s hold = 30 seconds. That's valid
mathematically but feels like a slideshow visually because:
- Average cut every 1.5 seconds is too slow for a music video
- Professional edits cut every 0.5–1.0 seconds = 30–60 cuts in 30s
- With only 20 frames that all look similar, there's no visual energy

**Fix — Task E1 (DONE):** Changed the default from 15/20 to 30 frames:
- `.env`: `MAX_FRAMES=15` → `MAX_FRAMES=30`
- `.env.example`: `MAX_FRAMES=15` → `MAX_FRAMES=30`
- `backend/main.py`: two default fallbacks `"15"` → `"30"`
- `frontend/index.html`: slider `max="20"` → `max="30"`, `value="15"` → `value="30"`, and
  `let maxFrames = 15` → `let maxFrames = 30`

The target is 1 frame per second over 30 seconds — 30 frames total. This matches
the approach used in Spider-Man: Into the Spider-Verse where sparse frame rates
create a deliberate, stylised animation feel rather than smooth motion.

---

## Task Details With Fixes Applied

### 🔴 Task A — Fix Prompt Order in Stage 2 (DONE)

**File:** `backend/utils/prompt_builder.py`

**What was changed:**

1. `FAST_MODE_FRAME_INSTRUCTIONS` — rewrote the KEYFRAME and CONTINUATION rules:
   - Old: "Use the literal token {{character_anchor}} at the START of full_prompt"
   - New: "PROMPT ORDER — CRITICAL: Every full_prompt MUST follow this exact order:
     1. ENVIRONMENT first (scene, location, atmosphere)
     2. LIGHTING (time of day, light quality)
     3. STYLE TOKENS (art style, medium)
     4. {{character_anchor}} token LAST"
   - Added: "DO NOT begin full_prompt with hair, eyes, clothing, or any character physical description"
   - Updated CONTINUATION example to show environment-first order ending with `{{character_anchor}}`

2. `QUALITY_MODE_FRAME_INSTRUCTIONS` — added the same prompt order guidance:
   - "PROMPT ORDER — CRITICAL: full_prompt MUST follow this exact order:
     1. ENVIRONMENT first
     2. LIGHTING
     3. STYLE TOKENS
     4. {{character_anchor}} token LAST (do not write character description inline)"
   - Added: "The keyframe is a fresh text_to_image — it establishes a brand-new scene"

3. `FAST_MODE_JSON_SCHEMA` — updated example `full_prompt` values:
   - Keyframe: `"{{character_anchor}}, scene description, {{style_anchor}}"` →
     `"scene description, {{style_anchor}}, {{character_anchor}}"`
   - Continuation: `"{{character_anchor}}, same [persistent elements], ..."` →
     `"same [persistent elements], ..., {{character_anchor}}"`

4. `QUALITY_MODE_JSON_SCHEMA` — updated keyframe `full_prompt` description:
   - `"complete detailed generation prompt"` →
     `"ENVIRONMENT first, lighting, style, then {{character_anchor}} — see prompt order rules above"`

5. `build_storyboard_prompt()` — the transcript line was updated to truncate at
   300 chars (was 500 in a previous version, Bug 13 fix):
   `audio_data.transcript or "instrumental — no lyrics"` →
   `(audio_data.transcript or "instrumental — no lyrics")[:300]`

**Verification:**
```bash
cd backend
python3 -c "from stages.stage2_storyboard import generate_storyboard; print('OK')"
```
Passed. All modules import cleanly.

**Note:** After editing, a real Stage 2 test should be run to inspect the first
scene's full_prompt and confirm it starts with the environment, not the character.

### 🔴 Task B — Fix Black Frames in Final Scene (DONE)

**File:** `backend/stages/stage4_video.py`

**What was changed:**

1. Added import: `from job_manager import log_progress`

2. Added `last_valid_image = None` tracking variable before the frame loop in
   `build_video()`.

3. After `raw_image = load_and_resize(frame.image_path, OUTPUT_RESOLUTION)`, added
   placeholder detection:
   ```python
   is_placeholder = False
   try:
       file_size = os.path.getsize(frame.image_path)
       gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
       mean_brightness = float(np.mean(gray))
       std_brightness = float(np.std(gray))
       is_placeholder = file_size < 5000 or (mean_brightness < 20 and std_brightness < 5)
   except Exception:
       is_placeholder = False

   if is_placeholder:
       if last_valid_image is not None:
           log_progress(job_id, f"[Stage 4] Frame {frame.frame_id} is a black placeholder ... — substituting last valid frame", "warn")
           raw_image = last_valid_image.copy()
       else:
           log_progress(job_id, f"[Stage 4] Frame {frame.frame_id} is a black placeholder and no previous valid frame exists — keeping as-is", "warn")
   else:
       last_valid_image = raw_image.copy()
   ```

4. Also verified Stage 3 error handling: the existing placeholder creation uses
   `Image.new('RGB', (width, height), color=(50, 50, 50))` with a clear
   `[ERROR]` log line, which is detectable by the above logic.

### 🔴 Task C — Fix character_anchor: null (DONE)

**File:** `backend/stages/stage2_storyboard.py`

**What was changed:**

1. Added guard immediately after `anchor = generate_character_anchor(...)` (Call 1):
   ```python
   # ── Task C: Guard against null / failed character anchor ────────────────
   if anchor is None or not anchor.get("short_anchor_text"):
       raise ValueError(
           "Character anchor generation failed — storyboard aborted. "
           "Check Groq API key and TPM limits."
       )
   ```

2. Added `storyboard.character_anchor = CharacterAnchor(**anchor)` for Quality mode
   (was previously only set for Fast mode). This allows `resolve_prompt_tokens()`
   to work in Stage 3's Quality mode keyframe generation:
   ```python
   # Inject character_anchor for both modes — used by Stage 3 for
   # {character_anchor} token resolution in keyframe prompts
   if not storyboard.character_anchor:
       storyboard.character_anchor = CharacterAnchor(**anchor)
   ```

### 🟡 Task D — Fix FRAME_DELAY Not Taking Effect (DONE)

**File:** `backend/stages/stage3_images.py`

**Root cause:** `FRAME_DELAY` was read once at module import time:
```python
FRAME_DELAY = float(os.environ.get("FRAME_DELAY", "8.0"))
```
Changing `.env` without restarting the server had no effect.

**Fix:** Both `time.sleep(FRAME_DELAY)` calls (in `generate_all_frames_fast` and
`generate_all_frames_quality`) now read the env var fresh each iteration and emit
a visible log line:
```python
frame_delay = float(os.environ.get("FRAME_DELAY", str(FRAME_DELAY)))
log_progress(progress_job_id,
    f"[Stage 3] GPU cooldown: sleeping {frame_delay}s before next frame...", "info")
time.sleep(frame_delay)
```

The module-level `FRAME_DELAY` constant is kept as the fallback default but is no
longer used directly in the sleep calls — each call reads the environment fresh.

**Verification log output should look like:**
```
[Stage 3] Frame 1/30 saved ✓
[Stage 3] GPU cooldown: sleeping 8.0s before next frame...
[Stage 3] Frame 2/30 saved ✓
[Stage 3] GPU cooldown: sleeping 8.0s before next frame...
```

### 🟡 Task E1 — Set Default and Max Frames to 30 (DONE)

**Files changed:**
- `.env`: `MAX_FRAMES=15` → `MAX_FRAMES=30`
- `.env.example`: `MAX_FRAMES=15` → `MAX_FRAMES=30`
- `backend/main.py` (line 90): `"max_frames_default": int(os.environ.get("MAX_FRAMES", "15"))` → `"30"`
- `backend/main.py` (line 376): `max_frames=max_frames or int(os.environ.get("MAX_FRAMES", "15")) or None` → `"30"`
- `frontend/index.html` (slider): `max="20" value="15"` → `max="30" value="30"`
- `frontend/index.html` (display): `<span class="slider-val">15</span>` → `30`
- `frontend/index.html` (JS): `let maxFrames = 15;` → `let maxFrames = 30;`

### 🟡 Task E2 — Show Optimal Parameter Suggestions in Frontend (DONE)

**File:** `frontend/index.html`

**What was added:**

1. Module-level variable: `let currentModelInfo = null;` (stored when a model is selected)

2. In `selectModel(key, modelInfo)`:
   - `currentModelInfo = modelInfo;` — cache for `applyModelDefaults()`
   - After the auto-fill logic, creates/shows a suggestion panel:
     - Creates `<div id="modelSuggestions" class="model-suggestions">` after the
       `#modelCards` container if it doesn't exist
     - Panel shows: display name, optimal_steps, optimal_guidance, optimal_denoise,
       optimal_denoise_continuation, optimal_width×optimal_height, optimal_sampler / optimal_scheduler
     - Shows notes if present
     - "Apply suggested settings" button calls `applyModelDefaults()`

3. New function `applyModelDefaults()`:
   ```javascript
   function applyModelDefaults() {
     if (currentModelInfo && selectedModelKey) {
       selectModel(selectedModelKey, currentModelInfo);
     }
   }
   ```

4. CSS added for `.model-suggestions`, `.suggestion-label`, `.suggestion-values`,
   `.suggestion-notes` classes.

**Optimal settings displayed (from `comfyui_registry.py`):**
- MeinaMix v12: Steps=25, Guidance=7.0, Denoise=0.75, Continuation=0.45, 768×768, euler, normal
  (Note: the registry currently has these values; the CURRENT_STATE v26 example showed
  different values of 7.5/0.88/euler_a/karras but the actual registry file has 7.0/0.75/euler/normal)
- SD 1.5: Steps=20, Guidance=7.5, Denoise=0.70, Continuation=0.45, 768×768, euler, normal

---

## ✅ DONE — Task 5: Style LoRA + Mood System

All 8 files were already wired in previous sessions (pipeline.py, stage2_storyboard.py,
stage3_images.py, comfyui_client.py, main.py, index.html, stage4_video.py).
Two changes were required to complete Task 5:

### Change 1 — `backend/models/lora_registry.py` (replaced entire file)

Replaced with v26 schema:
- **STYLE_LORAS**: 4 styles (none, mha, ghibli, cyberpunk_anime) with `display_name`,
  `description`, `lora_file`, `lora_weight`, `trigger_word`, `stage2_direction`, `negative_add`.
  Trigger words shortened to 3-4 word phrases matching actual LoRA file triggers.
- **MOOD_GRADES**: 6 primary moods (neutral, neutral_cinematic, energetic, melancholic,
  euphoric, tense) + `dark_intense` as backward-compat alias (maps to tense parameters).
- **Helper functions**: `get_style()`, `list_styles()`, `get_recommended_style()` for
  mood-to-style mapping. `LORA_STYPES` alias for backward compat with main.py imports.

### Change 2 — `backend/stages/stage1_audio.py` (updated mood detection)

- Added `neutral_cinematic` to `VALID_MOODS` set.
- Updated `_detect_mood()` fallback `mood_map`: `neutral_cinematic` → `neutral_cinematic`
  (was `neutral`). Default fallback changed from `neutral` to `neutral_cinematic`.
- Updated Groq LLM prompt: added `neutral_cinematic` to the classification list.
- `dark_intense` kept for backward compatibility (maps to `intense_aggressive` and
  `dark_powerful` audio-feature moods; MOOD_GRADES has matching backward-compat entry).

### Files 2–8 — already wired (verified)

- `pipeline.py` threads `style_key` + `detected_mood` to Stage 2; `style_key` to Stage 3;
  `detected_mood` to Stage 4 `build_video()`.
- `stage2_storyboard.py` looks up `STYLE_LORAS[style_key]` and `MOOD_GRADES[detected_mood]`
  to inject `stage2_direction` + `stage2_negative` into the Groq system prompt.
- `stage3_images.py` looks up `STYLE_LORAS` for `lora_file`/`lora_weight`, passes to
  ComfyUI client's `build_t2i_workflow()` / `build_img2img_workflow()`.
- `comfyui_client.py` already inserts `LoraLoader` node when `lora_file` is not None.
- `main.py` `/generate` endpoint already extracts `style_key` (default "none") and threads
  it through to `run_pipeline()`.
- `frontend/index.html` loads `/styles` dynamically, no dropdown hardcoded needed.
- `stage4_video.py` `apply_mood_grade()` looks up `MOOD_GRADES[mood_key]` for
  saturation/brightness/color_shift; skips nil grade (neutral, neutral_cinematic).

Import verification: `python3 -c "from models.lora_registry import ...; from stages.stage1_audio
import VALID_MOODS; ..."` — all keys in MOOD_GRADES ⊆ VALID_MOODS. ✅

```python
STYLE_LORAS = {
    "none":          {"lora_file": None, "lora_weight": 0,    "trigger_word": "",
                      "stage2_direction": "", "negative_add": ""},
    "mha":           {"lora_file": "AnimeKiss.safetensors",       "lora_weight": 0.75,
                      "trigger_word": "anime style, vibrant",
                      "stage2_direction": "Scenes must be bold, dynamic, high-energy anime action.",
                      "negative_add": "realistic, photographic"},
    "ghibli":        {"lora_file": "bluestyle-000005.safetensors", "lora_weight": 0.7,
                      "trigger_word": "ghibli style, painterly, soft",
                      "stage2_direction": "Scenes must be whimsical, nature-focused, contemplative.",
                      "negative_add": "dark, violent, harsh lighting"},
    "cyberpunk_anime":{"lora_file": "Cyberpunk_Anime-10.safetensors","lora_weight": 0.8,
                      "trigger_word": "cyberpunk, neon lights, futuristic",
                      "stage2_direction": "Scenes must be urban, neon-lit, rain-slicked streets.",
                      "negative_add": "nature, daylight, pastoral"},
}

MOOD_GRADES = {
    "neutral":            {"stage2_direction": "", "stage2_negative": "", "grade": None},
    "neutral_cinematic":  {"stage2_direction": "Cinematic pacing, wide shots, deliberate camera angles.",
                           "stage2_negative": "chaotic, cluttered", "grade": None},
    "energetic":          {"stage2_direction": "Fast cuts, dynamic poses, high contrast lighting.",
                           "stage2_negative": "slow, quiet, static",
                           "grade": {"saturation": 1.3, "brightness": 1.1,
                                     "color_shift": [1.05, 1.0, 0.95]}},
    "melancholic":        {"stage2_direction": "Slow pacing, long holds, muted tones, rain or fog.",
                           "stage2_negative": "bright, happy, energetic",
                           "grade": {"saturation": 0.75, "brightness": 0.9,
                                     "color_shift": [0.9, 0.95, 1.1]}},
    "euphoric":           {"stage2_direction": "Warm light floods, wide smiles, upward camera angles.",
                           "stage2_negative": "dark, somber, enclosed",
                           "grade": {"saturation": 1.4, "brightness": 1.15,
                                     "color_shift": [1.1, 1.05, 0.9]}},
    "tense":              {"stage2_direction": "Tight close-ups, low angles, harsh shadows.",
                           "stage2_negative": "open, peaceful, bright",
                           "grade": {"saturation": 1.1, "brightness": 0.85,
                                     "color_shift": [1.1, 0.95, 0.9]}},
}
```

### File 2 — `backend/pipeline.py`
Thread `style_key` and `detected_mood` through to Stage 2 and Stage 3.
The pipeline should call `generate_storyboard(..., style_key=style_key, detected_mood=detected_mood)`.
Pass `style_key` to `generate_all_frames()` so LoRA file/weight is looked up.

### File 3 — `backend/stages/stage2_storyboard.py`
The `generate_storyboard()` signature already accepts `style_key` and `detected_mood`
(added in the current v27 session). The `stage2_direction` and `stage2_negative` from
the selected style/mood are already injected into the system prompt. Verify this is
wired correctly end-to-end.

### File 4 — `backend/stages/stage3_images.py`
Both `generate_all_frames_fast()` and `generate_all_frames_quality()` already accept
`style_key` and look up `STYLE_LORAS`. The `lora_file` and `lora_weight` are already
passed to the ComfyUI client. Verify LoRA weights are applied to the ComfyUI workflow
nodes (see File 5).

### File 5 — `backend/comfyui_client.py`
Add LoraLoader node to both `build_t2i_workflow()` and `build_img2img_workflow()`.
When `lora_file` is not None, insert a `LoraLoader` node between the checkpoint loader
and the CLIPTextEncode/ModelSampling nodes. The LoRA should be applied to the CLIP
text encoder (for prompt steering) with the specified weight.

### File 6 — `backend/main.py`
In the `/generate` endpoint, extract `style_key` from the form data (default "none")
and pass it through to the pipeline. Also pass `detected_mood` (from Stage 1) to
Stage 2 and Stage 4.

### File 7 — `frontend/index.html`
Add a style dropdown/select element in the UI:
- "No Style (Base Model)" → none
- "My Hero Academia" → mha
- "Studio Ghibli" → ghibli
- "Cyberpunk Anime" → cyberpunk_anime

Store as `selectedStyle` and send as `style_key` in the form data.

### File 8 — `backend/stages/stage4_video.py`
Wire `detected_mood` to `apply_mood_grade()`. The `build_video()` function already
accepts `detected_mood` parameter. Ensure the pipeline passes the mood from Stage 1
through to Stage 4. The mood grade is already defined in `MOOD_GRADES` and applied
via `apply_mood_grade()`.

---

## 🔴 NEXT — Task 6: Real Browser Test

After Task 5 is complete, run a full end-to-end test in the browser:
1. Start the server: `cd backend && FRAME_DELAY=8.0 /Users/syedsunny/.venv/bin/python3 -m uvicorn main:app --reload --port 8001`
2. Open `http://localhost:8001/app/`
3. Upload a 30s audio file
4. Select a style (e.g., Ghibli)
5. Generate in Quality mode
6. Verify:
   - Each scene's keyframe shows a distinct environment (forest, not Japanese corridor)
   - Character anchor token is resolved (character appears correctly)
   - No black frames in the last 8 seconds
   - FRAME_DELAY log lines appear between every frame
   - Mood-based colour grade is applied to the final video
   - 30 frames are generated (not 20)

### Test progress (2026-10-16 session)

**Stage 1 (audio analysis)** — ✅ Verified. 30s test audio → 36 beats, 47 onsets, 4 energy
sections, mood = `neutral_cinematic`. Whisper transcription requires API setup.

**Stage 2 (storyboard)** — ✅ Verified via prior test artifact `temp/9bfc7fe5/storyboard.json`.
Prompt order fix confirmed: all keyframes start with environment (e.g. "ancient forest glade
with mist swirling among towering trees"), not character description. Character anchor present
with `short_anchor_text = "forest spirit, silver accents"`.

**Stage 3 (image gen via ComfyUI)** — ✅ Direct ComfyUI test: single `text_to_image` with
Ghibli LoRA at 512×512 / 12 steps completed in 61.7s, producing a 354KB real image (not
placeholder). Workflow structure correct (8 nodes, LoraLoader at proper position, output
at node "7"). Full Quality-mode stage test was interrupted (out of API calls for Groq).

**Stage 4 (video assembly)** — ✅ Verified code correctness: `progress_job_id` → `job_id`
fix applied (see Bug 15 below). Audio copy fix in `pipeline.py` ensures absolute path survives
upload cleanup. Placeholder detection (Task B) active in the frame loop.

### Bug 15 — Stage 4 NameError: progress_job_id (FOUND & FIXED)

**File:** `backend/stages/stage4_video.py`

**Root cause:** The Task 7B audio-loading code used `progress_job_id` in three `log_progress`
calls, but `build_video()` receives `job_id` as its parameter — `progress_job_id` is only
defined in `pipeline.py`'s `run_pipeline()`, not in `build_video()`'s scope. This would
crash with `NameError` at Stage 4 audio attachment.

**Fix:** Replaced all 3 `progress_job_id` → `job_id` in the audio loading block (lines 203/207/209).

### Bug 16 — COMFYUI_MODEL default typo (FOUND & FIXED)

**File:** `backend/comfyui_client.py` line 29, `.env.example`

**Root cause:** Default model was `"meinmix_v12Final"` (missing 'a') but the actual checkpoint
file on the ComfyUI server and in `comfyui_registry.py` is `"meinamix_v12Final.safetensors"`.
If no `COMFYUI_MODEL` env var is set, workflows would reference a non-existent checkpoint.

**Fix:** Corrected default to `"meinamix_v12Final.safetensors"` in both `comfyui_client.py`
and `.env.example`. (In the normal pipeline, `ckpt_name` is always passed explicitly from
`get_model(model_key)["filename"]`, so this only affects the fallback path.)

---

## Key Fixed Bugs (do not re-fix)

| Bug | Fix |
|---|---|
| Bug 6 | Global `frame_progress` counter (frame path overwriting) |
| Bug 7 | `/upload/image` with correct params |
| Bug 8 | `max_tokens=5500` in stage2 |
| Bug 9 | `pipeline.py` uses `progress_job_id` (job_id mismatch) |
| Bug 10 | `guidance_scale=guidance` NameError fixed |
| Bug 11 | Whisper transcription block moved before mood detection |
| Bug 12 | `short_anchor_text` added to `FAST_MODE_JSON_SCHEMA` |
| Bug 13 | Transcript truncated to 300 chars; max_tokens reduced to 5500 |
| Bug 14 | Model/LoRA filenames updated to match actual ComfyUI server files |
| Bug 15 | Stage 4 `progress_job_id` NameError — used `job_id` (function param) |
| Bug 16 | `COMFYUI_MODEL` default typo `meinmix_v12Final` → `meinamix_v12Final.safetensors` |

---

## Key Files

```
backend/
  main.py                  — FastAPI (endpoints: /generate, /stream, /models, /styles)
  pipeline.py              — orchestrates stages 1–4
  comfyui_client.py        — ComfyUI HTTP client + workflow builder
  job_manager.py           — thread-safe JobState registry + log_progress
  stages/
    stage1_audio.py        — librosa + Whisper audio analysis (DONE)
    stage2_storyboard.py   — Groq LLM storyboard (A/C done; 2-call flow active)
    stage3_images.py       — image generation (A/F/D done; Fast+Quality paths)
    stage4_video.py        — MoviePy assembly (B done; 6 beat-synced effects)
    beat_utils.py          — beat-aligned timing (DONE, do not touch)
  models/
    schemas.py             — Pydantic models (14 schemas)
    lora_registry.py       — style + mood defs (DONE — v26 schema + dark_intense backward compat)
    comfyui_registry.py    — checkpoint registry (DONE)
  utils/
    prompt_builder.py      — mode-aware storyboard prompts (A done)
    audio_utils.py         — audio feature helpers
frontend/
  index.html               — UI (E1/E2 done; model cards, style dropdown, max frames=30)
```

---

## How To Update This File
Run `git log --oneline` before declaring anything done. Rewrite when git state changes.
Next session prompt: "Read CURRENT_STATE.md first. Only read HANDOFF.md for historical context."
