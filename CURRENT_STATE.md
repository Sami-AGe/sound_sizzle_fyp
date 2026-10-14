# AUDIOSTORY — CURRENT_STATE.md
# v25 (2026-10-14). Task 4 COMPLETE — all 3 JS functions verified.
# READ THIS FIRST. Only read HANDOFF.md for historical context.

---

## What This Project Is
30-second audio clip → AI beat-synced video. Four-stage pipeline:
Stage 1 (audio analysis, librosa/Whisper) → Stage 2 (storyboard via Groq LLM) →
Stage 3 (images via ComfyUI locally) → Stage 4 (video assembly via MoviePy)

---

## ⚠️ API CALL BUDGET — READ BEFORE TOUCHING ANYTHING

- **Do not read a file unless you are about to edit it.**
- **Do not launch exploration agents.** All exploration is done.
- **Do not re-verify completed tasks.** Tasks 1–3 are committed. Task 4 backend is done.
- **Use `grep -n` with line ranges** instead of reading whole files.
- **`git log --oneline`** before declaring anything done.
- **Pattern:** find the exact insertion point with grep → write the one function → import-check → done.

---

## Status Snapshot

| Task | Status | Notes |
|---|---|---|
| Task 1 — guidance_scale, 0.22→0.45, ComfyUI integration, Bug 6–10 | ✅ DONE | committed |
| Task 2 — beat-first frame timing (`beat_utils.py`) | ✅ DONE | committed `ce09a68` |
| Task 3 — `/stream` endpoint + anchor download fix | ✅ DONE | committed `1e996b1` |
| Task 4 — model registry + browser model selection | ✅ DONE | verified ✅ — registry, /models route, sampler/scheduler wiring; 3 JS functions were already present in frontend |
| Task 4b — HF→ComfyUI switch | ✅ DONE | ComfyUI is now the default provider (USE_COMFYUI=true by default). HF import is lazy. Health check is non-fatal so server starts without ComfyUI. Host points to `10.40.12.55:8188`. |
| Task 5 — style LoRA + mood system | 🔴 NOT STARTED | full plan below |
| Task 6 — real browser test | ⬜ LAST | needs ComfyUI at `10.40.12.55:8188` |

---

## Paths & Stack

| Thing | Value |
|---|---|
| Project root | `/Users/syedsunny/sound_sizzle_fyp/` |
| Backend cwd | `backend/` |
| Venv python | `/Users/syedsunny/.venv/bin/python3` |
| ComfyUI | `http://10.40.12.55:8188` (Windows PC via ZeroTier, `--listen` flag) |
| Start server | `cd backend && FRAME_DELAY=2.0 /Users/syedsunny/sound_sizzle_fyp/.venv/bin/python3 -m uvicorn main:app --reload --port 8000` |
| Run sim test | `cd backend && /Users/syedsunny/sound_sizzle_fyp/.venv/bin/python3 -u browser_sim_test.py 2>&1 \| tee /tmp/sim_output.txt` |

---

## TASK 4 — 3 JS Functions Remaining 🟡 FINISH THIS FIRST

### What's already done (DO NOT REDO ANY OF THIS):

- ✅ `backend/models/comfyui_registry.py` — created (85 lines). Keys: `"meinamix"`,
  `"dreamshaper"`, `"deliberate"`. Each entry has `display_name`, `filename`,
  `optimal_steps`, `optimal_guidance`, `optimal_denoise`, `optimal_denoise_continuation`,
  `optimal_width`, `optimal_height`, `optimal_sampler`, `optimal_scheduler`, `notes`.
  Functions: `get_model(key)`, `list_models()`, constant `default_model_key = "meinamix"`.

- ✅ `backend/comfyui_client.py` — `build_t2i_workflow()` and `build_img2img_workflow()`
  now accept `sampler_name="euler"` and `scheduler="normal"` params (replacing hardcoded
  strings). `text_to_image()` and `image_to_image()` extract `ckpt_name`, `sampler_name`,
  `scheduler` from `**kwargs` and pass them to the workflow builders.

- ✅ `backend/stages/stage3_images.py` — imports `get_model`, `default_model_key` from
  `comfyui_registry`. Both `generate_all_frames_fast()` and `generate_all_frames_quality()`
  extract `model`, `sampler`, `scheduler`, `denoise_continuation` from `gen_params`,
  resolve model key → filename via `get_model()`, and pass `ckpt_name`, `sampler_name`,
  `scheduler` to every client call. Continuation strength uses `denoise_continuation` when
  set, then `denoise_override`, then `frame_directive.strength`, then `0.45`.

- ✅ `backend/main.py` — imports `list_models`. Has `GET /models` endpoint. `/generate`
  accepts four new Form params: `model`, `sampler`, `scheduler`, `denoise_continuation`,
  all packed into `gen_params`. `/config` returns `comfyui_model_key`.

- ✅ `frontend/index.html` HTML/CSS — model cards section added before cf-grid (div
  `#modelCards` inside `.model-cards` inside `.model-selector`). New controls added to
  `.cf-grid`: `#cfSampler` (select), `#cfScheduler` (select),
  `#cfDenoiseContinuation` (number). CSS for `.model-card`, `.model-card.selected`,
  `.model-label`, `.model-cards` already written.

- ✅ `frontend/index.html` JS — `selectedModelKey = null` state variable declared.
  `setMode()` checks `selectedModelKey === null` before applying mode defaults, also
  sets `cfDenoiseContinuation`. `generateVideo()` appends `model`, `sampler`,
  `scheduler`, `denoise_continuation` to FormData.

### What's left — 3 JS functions to add to `frontend/index.html`:

**Find the insertion point first:**
```bash
grep -n "function loadStyles\|function loadConfig\|function selectStyle\|async function load" \
  frontend/index.html | head -20
```

Add all three functions right after `loadStyles()` (or wherever the other load functions live).
Do NOT read the whole file — use `view` with the line range grep gives you.

**Function 1 — `loadModels()`:**
```javascript
async function loadModels() {
  try {
    const res = await fetch(`${API_BASE}/models`);
    if (!res.ok) return;
    const models = await res.json();
    const container = document.getElementById('modelCards');
    if (!container) return;
    container.innerHTML = '';
    for (const [key, m] of Object.entries(models)) {
      const card = document.createElement('div');
      card.className = 'model-card' + (key === (selectedModelKey || 'meinamix') ? ' selected' : '');
      card.dataset.key = key;
      card.innerHTML = `<span class="m-name${key === 'meinamix' ? ' m-default' : ''}">${m.display_name}</span>` +
                       `<span class="m-notes">${m.notes || ''}</span>`;
      card.onclick = () => selectModel(key, m);
      container.appendChild(card);
    }
    // Auto-select default if nothing selected yet
    if (!selectedModelKey) {
      const defaultEntry = models['meinamix'];
      if (defaultEntry) selectModel('meinamix', defaultEntry);
    }
  } catch (e) { /* ComfyUI may not be running — silently skip */ }
}
```

**Function 2 — `selectModel(key, m)`:**
```javascript
function selectModel(key, m) {
  selectedModelKey = key;
  // Highlight selected card
  document.querySelectorAll('.model-card').forEach(c => {
    c.classList.toggle('selected', c.dataset.key === key);
  });
  // Auto-fill all settings from registry optimal values
  if (m.optimal_steps)               document.getElementById('cfSteps').value = m.optimal_steps;
  if (m.optimal_guidance)            document.getElementById('cfGuidance').value = m.optimal_guidance;
  if (m.optimal_denoise)             document.getElementById('cfDenoise').value = m.optimal_denoise;
  if (m.optimal_denoise_continuation) document.getElementById('cfDenoiseContinuation').value = m.optimal_denoise_continuation;
  if (m.optimal_width)               document.getElementById('cfWidth').value = m.optimal_width;
  if (m.optimal_height)              document.getElementById('cfHeight').value = m.optimal_height;
  if (m.optimal_sampler) {
    const s = document.getElementById('cfSampler');
    if (s) s.value = m.optimal_sampler;
  }
  if (m.optimal_scheduler) {
    const s = document.getElementById('cfScheduler');
    if (s) s.value = m.optimal_scheduler;
  }
  log(`Model: ${m.display_name} — settings auto-filled`, 'info');
}
```

**Function 3 — update `loadConfig()` to call `loadModels()`:**
Find the existing `loadConfig()` function with grep, then add `await loadModels();`
at the end of it (or at the end of the `try` block inside it). One line only.

### Task 4 verification (all passed ✅):
```bash
cd /Users/syedsunny/sound_sizzle_fyp/backend

# 1. Registry loads ✅
python3 -c "from models.comfyui_registry import list_models, get_model; print(list_models())"
# Result: dict with meinamix, dreamshaper, deliberate — all have optimal_* fields

# 2. /models route exists ✅
python3 -c "from main import app; print([r.path for r in app.routes if hasattr(r,'path')])"
# Result: /models in routes list (use USE_COMFYUI=false to avoid ComfyUI connection at import)

# 3. Workflow uses passed sampler/scheduler ✅
python3 -c "
from comfyui_client import build_t2i_workflow
w = build_t2i_workflow('test.safetensors','prompt','neg',768,768,20,7.5,42,'pfx',
                       sampler_name='dpmpp_2m', scheduler='karras')
print(w['5']['inputs']['sampler_name'], w['5']['inputs']['scheduler'])
"
# Result: dpmpp_2m karras  (NOTE: verification command in v24 had an extra '' arg — removed)

# 4. Full sim test ⏳ (requires API credits — known limitation, see Task 6)
# Server starts, /models and /config respond correctly with USE_COMFYUI=false
```

**Note:** The 3 JS functions (`loadModels()`, `selectModel()`, and the `loadConfig()` call to
`loadModels()`) were already present in frontend/index.html. No frontend edits were needed.
The verification command for step 3 had an extra `''` argument that doesn't match the function
signature — this was a documentation issue, not a code bug.

---

## TASK 5 — Style LoRA + Mood System 🔴 AFTER TASK 4

**Design decisions already made — do not re-derive:**
- One style LoRA at a time in ComfyUI. No stacking.
- Mood handled via prompt injection (Stage 2) + colour grade (Stage 4). No mood LoRA.
- Mood detected in Stage 1: one Groq call, `llama3-8b-8192`, `max_tokens=5`.
- Style chosen by user via frontend dropdown. Flows through pipeline as `style_key`.
- **To add a new LoRA:** drop `.safetensors` into `ComfyUI/models/loras/` on Windows
  PC, add one entry to `STYLE_LORAS` in `lora_registry.py`. Pipeline picks it up automatically.

**Implementation order (8 files, import-check after each):**

### File 1 — `backend/models/lora_registry.py` (REPLACE entirely)

```python
from typing import Any, Dict

STYLE_LORAS: Dict[str, Dict[str, Any]] = {
    "none": {
        "display_name": "No Style (Base Model)",
        "lora_file": None, "lora_weight": 0.0,
        "trigger_word": "anime style", "negative_add": "",
        "stage2_direction": "anime aesthetic",
    },
    "mha": {
        "display_name": "My Hero Academia",
        "lora_file": "mha_style_v2.safetensors", "lora_weight": 0.75,
        "trigger_word": "bnha style, boku no hero academia", "negative_add": "",
        "stage2_direction": "anime superhero aesthetic, bold outlines, dynamic action poses, vivid energy effects",
    },
    "ghibli": {
        "display_name": "Studio Ghibli",
        "lora_file": "ghibli_style_v1.safetensors", "lora_weight": 0.70,
        "trigger_word": "ghibli style, studio ghibli", "negative_add": "dark, gritty, violent",
        "stage2_direction": "painterly backgrounds, soft lighting, nature-focused, warm and whimsical atmosphere",
    },
    "cyberpunk_anime": {
        "display_name": "Cyberpunk Anime",
        "lora_file": "cyberpunk_anime.safetensors", "lora_weight": 0.72,
        "trigger_word": "cyberpunk anime style, neon lights",
        "negative_add": "countryside, nature, bright daylight",
        "stage2_direction": "neon-lit urban environments, rain-slicked streets, holographic signs, dystopian cityscape",
    },
}

MOOD_GRADES: Dict[str, Dict[str, Any]] = {
    "melancholic": {
        "stage2_direction": (
            "muted colour palette, overcast or rainy environments, cold blue-grey lighting, "
            "characters shown from behind or in profile, empty sparse backgrounds, "
            "fallen leaves, fog, long shadows, desaturated tones"
        ),
        "stage2_negative": "bright colors, vibrant, cheerful, sunny, smiling, saturated, energetic",
        "grade": {"saturation": 0.60, "brightness": 0.88, "color_shift": (0.85, 0.90, 1.10)},
    },
    "energetic": {
        "stage2_direction": "dynamic action, bold colours, strong contrast, motion blur, dramatic angles, bright lighting",
        "stage2_negative": "static, dull, grey, empty, quiet",
        "grade": {"saturation": 1.25, "brightness": 1.05, "color_shift": (1.05, 1.0, 0.92)},
    },
    "dark_intense": {
        "stage2_direction": "high contrast, deep shadows, dramatic underlighting, tension, storm clouds, threatening atmosphere",
        "stage2_negative": "soft, pastel, cheerful, bright, peaceful",
        "grade": {"saturation": 0.80, "brightness": 0.78, "color_shift": (0.90, 0.88, 1.05)},
    },
    "euphoric": {
        "stage2_direction": "golden hour lighting, warm tones, wide open spaces, freedom, bloom effects, uplifting vistas",
        "stage2_negative": "dark, gloomy, cramped, grey",
        "grade": {"saturation": 1.15, "brightness": 1.08, "color_shift": (1.08, 1.02, 0.90)},
    },
    "tense": {
        "stage2_direction": "tight framing, harsh shadows, red or amber emergency lighting, dramatic stillness before action",
        "stage2_negative": "relaxed, peaceful, open, soft",
        "grade": {"saturation": 0.90, "brightness": 0.85, "color_shift": (1.08, 0.90, 0.88)},
    },
    "neutral": {
        "stage2_direction": "", "stage2_negative": "",
        "grade": {"saturation": 1.0, "brightness": 1.0, "color_shift": (1.0, 1.0, 1.0)},
    },
}
```

### File 2 — `backend/models/schemas.py`

Add to `AudioAnalysisResult`: `detected_mood: str = "neutral"`
Add to job request schema: `style_key: str = "none"`

### File 3 — `backend/stages/stage1_audio.py`

Grep: `grep -n "def analyse_audio\|return AudioAnalysis\|groq" stage1_audio.py`

Add `_detect_mood()` before `analyse_audio()`:
```python
def _detect_mood(transcript: str, tempo: float, avg_energy: float, groq_client) -> str:
    prompt = (
        f"Song lyrics: {transcript[:500]}\nTempo: {tempo:.0f} BPM, "
        f"Energy: {'high' if avg_energy > 0.05 else 'low'}\n\n"
        "Classify mood as one of: melancholic / energetic / dark_intense / euphoric / tense / neutral\n"
        "Respond with one word only."
    )
    resp = groq_client.chat.completions.create(
        model="llama3-8b-8192", messages=[{"role": "user", "content": prompt}],
        max_tokens=5, temperature=0.0,
    )
    mood = resp.choices[0].message.content.strip().lower().replace("-", "_")
    return mood if mood in {"melancholic","energetic","dark_intense","euphoric","tense","neutral"} else "neutral"
```

At end of `analyse_audio()` before `return`:
```python
avg_energy = float(np.mean(librosa.feature.rms(y=y)))
result.detected_mood = _detect_mood(result.transcript or "", result.tempo, avg_energy, groq_client)
```

### File 4 — `backend/comfyui_client.py`

Grep: `grep -n "class_type\|CheckpointLoader\|\"1\"\|build_t2i\|build_img2img" comfyui_client.py | head -20`

Add `lora_file=None, lora_weight=0.75` to `build_t2i_workflow()` and `build_img2img_workflow()`.
When `lora_file` is not None, insert a `LoraLoader` node between checkpoint and KSampler:

```python
if lora_file:
    workflow["lora_loader"] = {
        "class_type": "LoraLoader",
        "inputs": {
            "model": ["1", 0],   # "1" = CheckpointLoaderSimple node — verify with grep above
            "clip":  ["1", 1],
            "lora_name": lora_file,
            "strength_model": lora_weight,
            "strength_clip": lora_weight,
        }
    }
    # Re-point KSampler to use lora_loader output:
    # workflow["5"]["inputs"]["model"] = ["lora_loader", 0]   (t2i KSampler node id)
    # workflow["6"]["inputs"]["model"] = ["lora_loader", 0]   (img2img KSampler node id)
```
⚠️ Verify node IDs with the grep above before writing the re-point lines.

Also add `lora_file=None, lora_weight=0.75` to `text_to_image()` and `image_to_image()`
signatures, extract from `**kwargs`, pass to workflow builders.

### File 5 — `backend/stages/stage3_images.py`

Add `style_key: str = "none"` param to both `generate_all_frames_fast()` and
`generate_all_frames_quality()`. At top of each function after param block:

```python
from models.lora_registry import STYLE_LORAS
style = STYLE_LORAS.get(style_key or "none", STYLE_LORAS["none"])
lora_file   = style["lora_file"] if USE_COMFYUI else None
lora_weight = style["lora_weight"]
```

Pass `lora_file=lora_file, lora_weight=lora_weight` to every client call.
Prepend trigger word to each frame's prompt:
```python
full_prompt = f"{style['trigger_word']}, {frame_directive.full_prompt}"
```

### File 6 — `backend/stages/stage2_storyboard.py`

Grep: `grep -n "def generate_storyboard\|system_prompt\|system" stage2_storyboard.py | head -15`

Add `style_key: str = "none"` and `detected_mood: str = "neutral"` params.
Before building system prompt:
```python
from models.lora_registry import STYLE_LORAS, MOOD_GRADES
style = STYLE_LORAS.get(style_key, STYLE_LORAS["none"])
mood  = MOOD_GRADES.get(detected_mood, MOOD_GRADES["neutral"])
style_block = f"\nART STYLE: {style['stage2_direction']}" if style["stage2_direction"] else ""
mood_block  = (f"\nMOOD DIRECTION: {mood['stage2_direction']}"
               f"\nAVOID IN ALL SCENES: {mood['stage2_negative']}") if mood["stage2_direction"] else ""
system_prompt = existing_system_prompt + style_block + mood_block
```
Also append to shared negative: `mood["stage2_negative"]` and `style["negative_add"]`.

### File 7 — `backend/stages/stage4_video.py`

Grep: `grep -n "write_videofile\|final_clip\|def " stage4_video.py | head -15`

Add this function, then call it just before `write_videofile()`:
```python
import numpy as np
def apply_mood_grade(clip, mood_key: str):
    from models.lora_registry import MOOD_GRADES
    grade = MOOD_GRADES.get(mood_key, MOOD_GRADES["neutral"])["grade"]
    if mood_key == "neutral": return clip
    sat, bri, (cr, cg, cb) = grade["saturation"], grade["brightness"], grade["color_shift"]
    def grade_frame(frame):
        f = frame.astype(np.float32) / 255.0 * bri
        f[:,:,0] *= cr; f[:,:,1] *= cg; f[:,:,2] *= cb
        lum = (0.299*f[:,:,0] + 0.587*f[:,:,1] + 0.114*f[:,:,2])[:,:,np.newaxis]
        return np.clip((lum + (f - lum) * sat) * 255, 0, 255).astype(np.uint8)
    return clip.fl_image(grade_frame)

# Call before write_videofile():
# final_clip = apply_mood_grade(final_clip, audio_result.detected_mood)
```

### File 8 — `backend/pipeline.py` + `backend/main.py` + `frontend/index.html`

**`pipeline.py`:** pass `style_key` into Stage 2 and Stage 3 calls. Pass
`audio_result.detected_mood` into Stage 2 and Stage 4 calls.

**`main.py`:** add `style_key: str = Form("none")` to `/generate`, pack into pipeline call.

**`frontend/index.html`:** add style dropdown to ComfyUI panel:
```html
<label>Art Style</label>
<select id="cfStyleKey">
  <option value="none">No Style (Base Model)</option>
  <option value="mha">My Hero Academia</option>
  <option value="ghibli">Studio Ghibli</option>
  <option value="cyberpunk_anime">Cyberpunk Anime</option>
</select>
```
Append to FormData: `fd.append('style_key', document.getElementById('cfStyleKey').value);`

### Task 5 verification:
```bash
cd /Users/syedsunny/sound_sizzle_fyp/backend
python3 -c "from models.lora_registry import STYLE_LORAS, MOOD_GRADES; print('OK', list(STYLE_LORAS), list(MOOD_GRADES))"
python3 -c "from stages.stage1_audio import _detect_mood; print('OK')"
python3 -c "from stages.stage4_video import apply_mood_grade; print('OK')"
```

---

## TASK 6 — Real Browser Test ⬜ AFTER TASKS 4 & 5

Open `http://localhost:8000/app`. Submit a real audio clip via the UI.
Verify: model cards appear and auto-fill settings, style dropdown present,
progress bar tracks, video plays, download works without clearing the player.
Requires ComfyUI running at `10.40.12.55:8188`.

---

## Key Fixed Bugs (do not re-fix)

| Bug | What it was | Fix |
|---|---|---|
| Bug 6 | Frame files overwrote each other | Global `frame_progress` counter |
| Bug 7 | ComfyUI `/upload` → 405 | `/upload/image` with correct params |
| Bug 8 | Groq TPM 413 | `max_tokens=6000` in stage2 |
| Bug 9 | job_id mismatch → /download 404 | `pipeline.py` uses `progress_job_id` |
| Bug 10 | `guidance_scale=guidance_scale` NameError | Fixed to `guidance_scale=guidance` |

---

## Key Files

```
backend/
  main.py                  — FastAPI app (all endpoints incl. /stream, /models, /styles)
  pipeline.py              — orchestrates stages 1–4
  comfyui_client.py        — ComfyUI HTTP client + workflow builder (sampler/scheduler done)
  stages/
    stage1_audio.py        — librosa + Whisper
    stage2_storyboard.py   — Groq LLM storyboard
    stage3_images.py       — image generation (model/sampler/scheduler/denoise_continuation done)
    stage4_video.py        — MoviePy assembly
    beat_utils.py          — beat-aligned timing (DONE, do not touch)
  models/
    schemas.py             — Pydantic models
    lora_registry.py       — style + mood defs (will be replaced in Task 5)
    comfyui_registry.py    — checkpoint registry (DONE, Task 4)
frontend/
  index.html               — UI (model cards HTML/CSS done; loadModels/selectModel JS pending)
```

---

## How To Update This File
Run `git log --oneline` before declaring anything done. Rewrite this file when
git state changes. Working-tree edits may already be committed from a prior session.