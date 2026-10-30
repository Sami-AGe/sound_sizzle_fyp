"""
AudioStory — Prompt Builder

Builds image generation prompts from audio analysis data and storyboard directives.
For v2 dual-mode: build_storyboard_prompt() is mode-aware, generating different
frame instructions depending on whether Fast or Quality mode is selected.
"""

from models.schemas import AudioAnalysisResult
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────
# Mode-specific frame instruction blocks
# ─────────────────────────────────────────────────────────────────────────

FAST_MODE_FRAME_INSTRUCTIONS = """
GENERATION MODE: FAST (text2img only)

Since there is no img2img, YOU are responsible for visual consistency.
The image model cannot see previous frames. YOU must describe them in text.

VISUAL LANGUAGE CHEAT SHEET — use ONLY these concrete terms:
- Camera angle: "low angle" (up at subject from below), "high angle" (down at subject from above), "eye level"
- Shot distance: "wide shot" (full body + environment), "medium shot" (waist up), "close-up" (face fills frame), "extreme close-up"
- Character orientation: "facing camera", "profile left", "three-quarter view", "over-the-shoulder"
- Lighting: "single key light from upper left", "soft diffused light", "harsh rim light from behind"
- Movement: "character turning left", "raising arm", "stepping forward", "blinking", "nodding head", "hand to heart"

Rules for KEYFRAME prompts:
- PROMPT ORDER — CRITICAL: Every full_prompt MUST follow this exact order:
  1. ENVIRONMENT first (where is the character, what surrounds them — 10-15 words)
  2. LIGHTING (5-7 words): time of day, light quality, direction
  3. STYLE TOKENS (3-5 words): art style, medium, quality
  4. {{character_anchor}} token LAST (3-5 word identity tag, NOT full physical description)
- Example full_prompt: "misty forest clearing at dawn, soft golden mist, cool morning light from above, painterly illustration, studio ghibli style, {{character_anchor}}"
- DO NOT begin full_prompt with hair, eyes, clothing, or character physical description
- Set visual_state: list ALL persistent elements (camera angle, shot distance, lighting, environment, character pose) for continuations

Rules for CONTINUATION prompts:
- Repeat EVERY persistent element from visual_state with "same" prefix, then add what changes
  Example: "same forest clearing, same soft golden mist, same medium shot, same character facing camera, character raising arm slowly, {{character_anchor}}"
- Then describe ONLY what changes — be specific, not abstract
  Good: "character blinks", "camera tilts up", "hand raises", "head turns left"
  Bad: "dynamic motion", "energy shift", "dramatic change"
- End with {{character_anchor}} last
- NO modifier_prompt field — only full_prompt
- NO strength field — not applicable

The locked character anchor text is:
{character_anchor_text}

The locked style anchor is:
{style_anchor}
"""

QUALITY_MODE_FRAME_INSTRUCTIONS = """
GENERATION MODE: QUALITY (img2img)

img2img handles visual inheritance via pixels. Your job per frame is lightweight.

VISUAL LANGUAGE CHEAT SHEET — use ONLY these concrete terms:
- Camera angle: "low angle", "high angle", "eye level"
- Shot distance: "wide shot", "medium shot", "close-up", "extreme close-up"
- Character: "facing camera", "profile left", "three-quarter view", "over-the-shoulder"
- Lighting: "single key light from upper left", "soft diffused light", "harsh rim light from behind"
- Movement: "character turning left", "raising arm", "stepping forward", "blinking", "nodding head"

Rules for KEYFRAME prompts:
- PROMPT ORDER — CRITICAL: full_prompt MUST follow this exact order:
  1. ENVIRONMENT first (scene setting, location, atmosphere, 10-15 words)
  2. LIGHTING (5-7 words): time of day, light quality, direction
  3. STYLE TOKENS (3-5 words): art style, medium, quality
  4. {{character_anchor}} token LAST (do not write character physical description —
    it is injected programmatically from the locked anchor)
- DO NOT begin full_prompt with hair, eyes, clothing, or character physical description
- subject_pose: concrete positioning like "standing, facing camera, weight on left leg"
- camera_angle: "low angle" or "high angle" or "eye level"
- lighting: "soft diffused light from above" or "harsh rim light from behind"
- The keyframe is a fresh text_to_image — it establishes a brand-new scene

Rules for CONTINUATION prompts:
- modifier_prompt: ONLY describe the specific change (2-12 words)
  Good: "character blinks slowly", "camera tilts up gently", "arm raises slightly", "head turns left"
  Bad: "more dynamic", "intense motion", "energy shift"
CONTINUATION FRAME STRENGTH RULES — CRITICAL:
Strength controls how much the image changes. Values below 0.50 are INVISIBLE in video.

  - Subtle motion (blink, breath, hair sway, light flicker): strength 0.50–0.60
  - Moderate motion (arm raise, head turn, camera pan, lighting shift): strength 0.65–0.75
  - Strong motion (body rotation, zoom, dramatic scene element change): strength 0.75–0.88
  - NEVER generate strength below 0.50 — values of 0.17–0.35 produce invisible changes
  - Each frame holds for ~1 second in the video — motion must be clearly visible at that pace
- what_changes: human-readable description for logging
- NO full_prompt on CONTINUATION frames
"""


# ─────────────────────────────────────────────────────────────────────────
# Mode-specific JSON schema templates (embedded in prompt for LLM guidance)
# ─────────────────────────────────────────────────────────────────────────

FAST_MODE_JSON_SCHEMA = '''
{
  "generation_mode": "fast",
  "main_story": "2-3 sentence overall narrative summary",
  "total_scenes": 6,
  "character_anchor": {
    "hair": "...",
    "eyes": "...",
    "clothing": "...",
    "body": "...",
    "distinguishing": "...",
    "full_anchor_text": "...",
    "short_anchor_text": "8-12 word description with exact colors, e.g. 'teal bob hair, amber eyes, black trench coat, pink circuit patterns'",
    "style_anchor": "...",
    "reference_prompt": "..."
  },
  "style_anchor": "style anchor text from above",
  "emotional_arc": {
    "intro": "visual/emotional description of intro section",
    "buildup": "visual/emotional description of buildup section",
    "climax": "visual/emotional description of climax section",
    "outro": "visual/emotional description of outro section"
  },
  "scenes": [
    {
      "scene_id": 1,
      "scene_seed": 42,
      "start_time": 0.0,
      "end_time": 8.5,
      "narrative": "what happens in this scene",
      "setting": "physical location / environment",
      "mood": "emotional tone",
      "color_palette": "color description",
      "transition_from_previous": "cut or fade or glitch_cut",
      "frames": [
        {
          "frame_id": 1,
          "timestamp": 0.0,
          "type": "KEYFRAME",
          "full_prompt": "scene description, {{style_anchor}}, {{character_anchor}}",
          "visual_state": "all persistent elements that continue into next frames",
          "subject_pose": "how character is positioned",
          "camera_angle": "wide shot or closeup etc",
          "lighting": "lighting description",
          "what_changes": "human readable description (optional)"
        },
        {
          "frame_id": 2,
          "timestamp": 0.5,
          "type": "CONTINUATION",
          "full_prompt": "same [persistent elements], [what changes], {{character_anchor}}",
          "what_changes": "what changed in this frame"
        }
      ]
    }
  ]
}
'''

QUALITY_MODE_JSON_SCHEMA = '''
{
  "generation_mode": "quality",
  "main_story": "2-3 sentence overall narrative summary",
  "total_scenes": 6,
  "subject_description": "overall character description",
  "subject_reference_prompt": "full prompt for generating a neutral standing pose reference image",
  "scenes": [
    {
      "scene_id": 1,
      "start_time": 0.0,
      "end_time": 8.5,
      "narrative": "what happens in this scene",
      "setting": "physical location / environment",
      "mood": "emotional tone",
      "color_palette": "color description",
      "transition_from_previous": "cut or fade or glitch_cut",
      "frames": [
        {
          "frame_id": 1,
          "timestamp": 0.0,
          "type": "KEYFRAME",
          "full_prompt": "ENVIRONMENT first, lighting, style, then {{character_anchor}} — see prompt order rules above",
          "subject_pose": "how character is positioned",
          "camera_angle": "camera angle description",
          "lighting": "lighting description",
          "what_changes": "optional"
        },
        {
          "frame_id": 2,
          "timestamp": 0.5,
          "type": "CONTINUATION",
          "modifier_prompt": "SHORT description of ONLY what changes",
          "strength": 0.65,
          "what_changes": "what changed"
        }
      ]
    }
  ]
}
'''


# ─────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────

def build_storyboard_prompt(
    audio_data: AudioAnalysisResult,
    anchor: dict,
    style_name: str,
    style_description: str,
    style_trigger_word: str,
    generation_mode: str,
    num_scenes: int,
    max_frames: int = None,
    unique_moments: list = None,
) -> str:
    """
    Builds the full prompt sent to Groq for storyboard generation (Call 2).
    Mode-aware: generates different JSON schema expectations for Fast vs Quality.
    """
    significant_beats = filter_significant_beats(
        audio_data.beat_times,
        audio_data.rms_values,
        audio_data.rms_timestamps,
        audio_data.onset_times
    )

    # Limit beats to reduce prompt size (avoid TPM limits)
    beat_info = _format_beats(significant_beats[:15])
    energy_sections = _format_energy_sections(audio_data.energy_sections)

    # ── Visual direction derived from audio features ─────────────────────
    energy_hint = derive_energy_visual_hint(audio_data)
    arc_hints = derive_arc_hints(audio_data)
    lyric_symbols = extract_lyric_symbols(audio_data.transcript)
    frames_per_scene = max(2, max_frames // num_scenes) if max_frames else 3

    # ── Mode-specific frame instructions ─────────────────────────────────
    if generation_mode == "fast":
        frame_instructions = FAST_MODE_FRAME_INSTRUCTIONS.format(
            character_anchor_text=anchor["full_anchor_text"],
            style_anchor=anchor["style_anchor"]
        )
        json_schema = FAST_MODE_JSON_SCHEMA
    else:
        frame_instructions = QUALITY_MODE_FRAME_INSTRUCTIONS
        json_schema = QUALITY_MODE_JSON_SCHEMA

    unique_moments_text = _format_unique_moments(unique_moments) if unique_moments else ""

    return f"""
You are creating a storyboard for a 30-second {style_name} music video.

══════════════════════════════════════════════
LOCKED CHARACTER ANCHOR (use verbatim in prompts)
══════════════════════════════════════════════
Character: {anchor["full_anchor_text"]}
Style anchor: {anchor["style_anchor"]}

For Fast mode prompts, use the literal token {{character_anchor}} where
the character anchor text goes, and {{style_anchor}} where the style anchor goes.
These tokens will be substituted programmatically.

══════════════════════════════════════════════
AUDIO ANALYSIS
══════════════════════════════════════════════
Duration: {audio_data.duration:.1f}s
Tempo: {audio_data.tempo:.1f} BPM
Overall mood: {audio_data.detected_mood}
Tonal brightness: {"bright and energetic" if audio_data.spectral_centroid_mean > 0.5 else "dark and heavy"}
Lyrics: {(audio_data.transcript or "instrumental")[:100]}

Energy sections:
{energy_sections}

Major onsets (drops/hits) at: {[round(t, 2) for t in audio_data.onset_times[:8]]}

══════════════════════════════════════════════
UNIQUE MOMENTS (keyframe candidates — most distinctive audio points)
══════════════════════════════════════════════
These timestamps have the highest combined novelty score (timbral change +
energy deviation + beat onset). Place your most visually dynamic scene at the
highest-scoring moment. Each scene boundary should snap to a nearby unique moment.

{unique_moments_text}

══════════════════════════════════════════════
AUDIO → VISUAL TRANSLATION RULES
══════════════════════════════════════════════
Apply these rules when designing scenes and frames.

{energy_hint}

CAMERA & FRAMING — Use concrete, specific terms:
- Camera angle: "low angle" (looking up at subject from below), "high angle" (looking down from above), "eye level" (viewer at subject height)
- Shot distance: "wide shot" (full body + environment), "medium shot" (waist up), "close-up" (face fills frame), "extreme close-up" (single feature like eyes)
- Character orientation: "facing camera", "profile left", "three-quarter view", "over-the-shoulder"
- Movement: "character turning left", "raising arm", "stepping forward", "nodding head", "hand to heart"

LIGHTING — Use specific descriptions:
- "single key light from upper left" (one bright light source)
- "soft diffused light" (even, shadowless, like overcast)
- "harsh rim light from behind" (backlight creates outline)
- "dappled sunlight through leaves" (patchy natural light)
- "neon glow from right side" (colored artificial light)

TRANSITION triggers:
- Onset at HIGH energy → transition_from_previous = "glitch_cut"
- Beat change without onset → transition_from_previous = "hard_cut"
- Low energy scene change → transition_from_previous = "dissolve"
- Major drop (loudest onset) → transition_from_previous = "flash_cut"
- If character changes pose → describe the pose change in what_changes

Color rules by mood:
- intense_aggressive / dark_powerful → deep reds, blacks, harsh contrasts, single harsh light source
- energetic_euphoric / uplifting_heroic → bright warm golds, whites, saturated colors, multiple light sources
- melancholic_lonely / peaceful_dreamy → desaturated blues, purples, soft gradients, fog/mist
- chaotic_anxious → mixed complementary colors, harsh shadows, broken light
- neutral_cinematic → cool neutrals, single key light, film-style contrast

Lyric symbolism to encode visually:
{lyric_symbols}

══════════════════════════════════════════════
EMOTIONAL ARC (map scenes to this)
══════════════════════════════════════════════
{arc_hints}

══════════════════════════════════════════════
FRAME DISTRIBUTION INSTRUCTIONS
══════════════════════════════════════════════
SCENE PACING RULES — CRITICAL:
- Target: 1 new scene (keyframe) every 3–5 seconds of audio
- For 30 seconds of audio: generate 6–8 scenes (NOT 3 scenes)
- Each scene: 1 KEYFRAME + 2–4 CONTINUATION frames (3–5 frames total per scene)
- More scenes with fewer frames = dynamic video that feels like a real edit
- Fewer scenes with many frames = slideshow that feels static and boring
- With 30 total frames across 6 scenes: ~5 frames per scene (1 keyframe + 4 continuations)
- With 30 total frames across 8 scenes: ~4 frames per scene (1 keyframe + 3 continuations)

Distribute {max_frames if max_frames else 'all'} total frames across {num_scenes} scenes using this pattern:
- First frame of each scene = KEYFRAME (fresh scene from scratch)
- 2-4 CONTINUATION frames after each KEYFRAME (not 1 — enough for smooth motion)
- CONTINUATION modifier_prompt describes ONLY the change: e.g. "character blinks", "camera tilts up", "rain intensifies", "arm raises slightly", "head turns left"
- Keyframes should be at the unique moments listed above (snap the highest-energy scene to the most distinctive moment)

========

FRAME GENERATION INSTRUCTIONS ({generation_mode.upper()} MODE)
══════════════════════════════════════════════
{frame_instructions}

══════════════════════════════════════════════
REQUIRED JSON OUTPUT SCHEMA (CRITICAL — follow EXACTLY)
══════════════════════════════════════════════
Return ONLY valid JSON. Every field listed below MUST be present. Use exact field names and types.

{json_schema}

══════════════════════════════════════════════
STRUCTURE REQUIREMENTS
══════════════════════════════════════════════
- Exactly {num_scenes} scenes
- Approximately {frames_per_scene} frames per scene (target {max_frames if max_frames else 'unlimited'} total)
- Each scene: 1 KEYFRAME at start, 2-4 CONTINUATIONS after it
- Keyframe timestamps should snap to the nearest unique moment above
- Scene boundaries MUST align with unique moments or onset timestamps
- Generation mode in output: "{generation_mode}" (lowercase)
- The "type" field for each frame MUST be exactly "KEYFRAME" or "CONTINUATION" (uppercase)
- Do NOT include main_story, total_scenes, or scene fields as nested — they are top-level required fields

Return ONLY valid JSON. Do not include any text before or after the JSON.
"""


# ─────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────

def _format_unique_moments(moments: list) -> str:
    """Formats unique moments list as readable text for the LLM prompt."""
    if not moments:
        return "No unique moments detected."
    lines = []
    for m in moments:
        lines.append(
            f"  t={m['timestamp']:.2f}s | energy={m['energy']:.2f} | "
            f"brightness={m['brightness']:.2f} | {m['description']}"
        )
    return "\n".join(lines)


def derive_energy_visual_hint(audio_data: AudioAnalysisResult) -> str:
    """Translates RMS energy profile into visual direction text for the LLM."""
    avg_e = sum(audio_data.rms_values) / len(audio_data.rms_values)
    peak_e = max(audio_data.rms_values)
    hints = []
    if peak_e > 0.85:
        hints.append("This audio has VERY high peak energy — at least one scene should use extreme close-ups, heavy saturation, and maximum visual intensity.")
    if avg_e > 0.6:
        hints.append("Overall high energy — prefer dynamic compositions, movement, diagonal lines over static balanced frames.")
    elif avg_e < 0.3:
        hints.append("Overall low energy — prefer static wide shots, muted palettes, minimal movement between continuation frames.")
    num_onsets = len(audio_data.onset_times)
    if num_onsets > 8:
        hints.append(f"High onset density ({num_onsets} hits detected) — expect frequent glitch transitions and high-impact scene cuts.")
    return "\n".join(hints) if hints else "Energy is moderate — balance dynamic and static elements."


def derive_arc_hints(audio_data: AudioAnalysisResult) -> str:
    """Maps audio energy sections to narrative arc guidance."""
    lines = []
    for s in audio_data.energy_sections:
        label = s["label"]
        energy = s["energy_label"]
        start = s["start"]
        end = s["end"]
        if label == "intro":
            visual = "Establishing shot, subdued palette, character introduced at rest or in motion toward something."
        elif label == "buildup":
            visual = "Tension rising, camera moves closer, color saturation increasing, character's emotion intensifying."
        elif label == "climax":
            visual = "Peak visual intensity, most saturated colors, most dynamic composition, character's defining moment."
        else:
            visual = "Resolution or reflection. Energy fading, wide shot, character at rest or moving away."
        lines.append(f"  [{label.upper()} | {start:.1f}s–{end:.1f}s | energy: {energy}]: {visual}")
    return "\n".join(lines)


def extract_lyric_symbols(transcript: Optional[str]) -> str:
    """Provides lyric symbolism instructions to the LLM storyboarder."""
    if not transcript or len(transcript.strip()) < 10:
        return "No lyrics detected. Drive visual symbolism entirely from audio mood and energy."
    return (
        f'Lyrics present: "{transcript[:300]}..."\n'
        "Instructions: Identify 2-3 key emotional images or metaphors from the lyrics and encode them as visual elements. "
        "Example: 'running from the dark' → scene lit from behind with character silhouette moving forward. "
        "'fire inside' → warm light source emanating from character. "
        "Do not illustrate lyrics literally — translate them into visual mood and composition."
    )


# ─────────────────────────────────────────────────────────────────────────
# Existing utilities (preserved)
# ─────────────────────────────────────────────────────────────────────────

def filter_significant_beats(
    beat_times: list, rms_values: list, rms_timestamps: list, onset_times: list
) -> list:
    """
    From all beat timestamps, extract only the most meaningful ones.
    Keeps onset beats + every regular beat with its energy value.
    """
    from utils.audio_utils import get_energy_at_time

    result = []
    for i, t in enumerate(beat_times):
        energy = get_energy_at_time(rms_values, rms_timestamps, t)
        is_onset = any(abs(t - o) < 0.1 for o in onset_times)
        result.append({
            "timestamp": round(t, 2),
            "energy": round(energy, 2),
            "is_onset": is_onset,
            "beat_index": i
        })
    return result


def _format_beats(beats: list) -> str:
    """Formats beat data as readable text for the prompt."""
    if not beats:
        return "No significant beats detected"
    lines = []
    for b in beats:
        onset_marker = " <<< ONSET" if b["is_onset"] else ""
        lines.append(f"  t={b['timestamp']:.2f}s, energy={b['energy']:.2f}{onset_marker}")
    return "\n".join(lines)


def _format_energy_sections(sections: list) -> str:
    """Formats energy sections as readable text for the prompt."""
    if not sections:
        return "No sections detected"
    lines = []
    for s in sections:
        lines.append(
            f"  {s['label']}: {s['start']:.1f}s-{s['end']:.1f}s, "
            f"energy={s['avg_energy']:.2f}, {s['energy_label']}"
        )
    return "\n".join(lines)
