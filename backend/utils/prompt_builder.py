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

Rules for KEYFRAME prompts:
- Use the literal token {{character_anchor}} at the START of full_prompt
- Describe setting, lighting, camera, pose in full detail
- End with {{style_anchor}}
- Set visual_state: describe ALL visual elements that will persist through this scene's continuations

Rules for CONTINUATION prompts:
- Use the literal token {{character_anchor}} at the START of full_prompt
- Re-state EVERY persistent element from visual_state using "same [element]" language
  Example: "same dark alley, same cold blue side lighting, same medium shot"
- Then describe what changes in this frame
- End with {{style_anchor}}
- NO modifier_prompt field — only full_prompt
- NO strength field — not applicable

The locked character anchor text is:
{character_anchor_text}

The locked style anchor is:
{style_anchor}

Example CONTINUATION prompt:
"{{character_anchor}}, same dark rain-slicked alley, same cold blue side lighting, same medium shot angle, rain intensifying around character, fists beginning to clench, {{style_anchor}}"
"""

QUALITY_MODE_FRAME_INSTRUCTIONS = """
GENERATION MODE: QUALITY (img2img)

img2img handles visual inheritance via pixels. Your job per frame is lightweight.

Rules for KEYFRAME prompts:
- full_prompt: complete detailed generation prompt including character description, setting, lighting, camera, style tags
- subject_pose, camera_angle, lighting as separate fields

Rules for CONTINUATION prompts:
- modifier_prompt: SHORT description of ONLY what changes from the previous frame
  (2-15 words — the model sees the previous frame's pixels, so don't re-describe everything)
- strength: float 0.15–0.45
  - 0.15–0.22: subtle (slight expression shift, lighting flicker)
  - 0.22–0.35: moderate (body movement, partial scene change)
  - 0.35–0.45: strong (significant pose change, major lighting shift)
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
  "total_scenes": "INTEGER (must match the number of scenes in the array below)",
  "character_anchor": {
    "hair": "...",
    "eyes": "...",
    "clothing": "...",
    "body": "...",
    "distinguishing": "...",
    "full_anchor_text": "...",
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
          "full_prompt": "{{character_anchor}}, scene description, {{style_anchor}}",
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
          "full_prompt": "{{character_anchor}}, same [persistent elements], [what changes], {{style_anchor}}",
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
  "total_scenes": "INTEGER (must equal num scenes below)",
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
          "full_prompt": "complete detailed generation prompt",
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
          "strength": 0.22,
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
    max_frames: int = None
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

    beat_info = _format_beats(significant_beats[:30])
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
Lyrics: {audio_data.transcript or "instrumental — no lyrics"}

Energy sections:
{energy_sections}

Major onsets (drops/hits) at: {[round(t, 2) for t in audio_data.onset_times[:8]]}

══════════════════════════════════════════════
AUDIO → VISUAL TRANSLATION RULES
══════════════════════════════════════════════
Apply these rules when designing scenes and frames:

{energy_hint}

Camera rules by energy:
- HIGH energy section → dynamic angles, diagonal compositions, ECU (extreme close-up) on face/hands
- MEDIUM energy → medium shots, slight angles, balanced composition
- LOW energy → wide establishing shots, static camera, centered composition, rule of thirds

Color rules by mood:
- intense_aggressive / dark_powerful → deep reds, blacks, harsh contrasts, single harsh light source
- energetic_euphoric / uplifting_heroic → bright warm golds, whites, saturated colors, multiple light sources
- melancholic_lonely / peaceful_dreamy → desaturated blues, purples, soft gradients, fog/mist
- chaotic_anxious → mixed complementary colors, harsh shadows, broken light
- neutral_cinematic → cool neutrals, single key light, film-style contrast

Transition rules by audio event:
- Onset at HIGH energy → transition_from_previous = "glitch_cut"
- Beat change without onset → transition_from_previous = "hard_cut"
- Low energy scene change → transition_from_previous = "dissolve"
- Major drop (loudest onset) → transition_from_previous = "flash_cut"

Lyric symbolism to encode visually:
{lyric_symbols}

══════════════════════════════════════════════
EMOTIONAL ARC (map scenes to this)
══════════════════════════════════════════════
{arc_hints}

══════════════════════════════════════════════
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
- Each scene: 1 KEYFRAME at start, rest are CONTINUATIONS
- Scene boundaries MUST align with onset timestamps or section boundaries
- Generation mode in output: "{generation_mode}" (lowercase)
- The "type" field for each frame MUST be exactly "KEYFRAME" or "CONTINUATION" (uppercase)
- Do NOT include main_story, total_scenes, or scene fields as nested — they are top-level required fields

Return ONLY valid JSON. Do not include any text before or after the JSON.
"""


# ─────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────

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
