"""
AudioStory — Stage 2: Storyboard Generation

Sends the audio analysis result to Groq's LLM and receives a structured
JSON storyboard that dictates the full visual plan for the video.

v2 dual-mode: Two sequential Groq calls.
  Call 1 — generate_character_anchor: locked character + style descriptor
  Call 2 — generate_storyboard: full scene + frame plan, mode-aware

The output is a machine-readable plan, not a prose paragraph.
"""

import json
import re
import os
import time

from groq import Groq
from models.schemas import AudioAnalysisResult, Storyboard, CharacterAnchor
from utils.prompt_builder import build_storyboard_prompt


# Initialize Groq client (API key from .env)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

# Recommended Groq model for structured JSON output
STORYBOARD_MODEL = "openai/gpt-oss-120b"

# Default system prompt
SYSTEM_PROMPT = (
    "You are a professional music video storyboarder and creative director. "
    "You generate precise structured JSON storyboards for beat-synced video edits. "
    "ONLY respond with valid JSON. No markdown, no explanation, no preamble. "
    "Raw JSON object only."
)


def generate_character_anchor(
    audio_data: AudioAnalysisResult,
    style_name: str,
    style_trigger_word: str,
    max_retries: int = 3
) -> dict:
    """
    Call 1: Dedicated Groq call that generates the locked character + style anchor.
    Short, focused, fast. Result is frozen for the entire pipeline run.

    Returns a dict with: hair, eyes, clothing, body, distinguishing,
    full_anchor_text, short_anchor_text, style_anchor, reference_prompt.

    Approach: Ask the LLM for only the short individual fields (hair, eyes, etc.),
    then construct full_anchor_text and reference_prompt programmatically.
    This avoids LLM truncation on the longer derived fields.
    """
    prompt = f"""
You are designing the visual identity for a {style_name} music video.

Audio mood: {audio_data.detected_mood}
Lyrics excerpt: {(audio_data.transcript or "instrumental")[:200]}

Generate a character for this video. Return ONLY this JSON (keep all field values short, 5-15 words each):

{{
  "hair": "precise hair description",
  "eyes": "eye description with emotional quality",
  "clothing": "specific clothing with color and detail",
  "body": "body type and build",
  "distinguishing": "one unique visual trait (energy aura, weapon, accessory, etc.)",
  "short_anchor_text": "3-5 word identity tag, e.g. 'green-haired hero, energy powers'"
}}
"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=STORYBOARD_MODEL,
                messages=[
                    {"role": "system", "content": "Return only valid JSON. No explanation. No markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=512
            )

            raw = strip_json_fences(response.choices[0].message.content.strip())
            try:
                anchor = json.loads(raw)
            except json.JSONDecodeError:
                fixed = fix_json(raw)
                anchor = json.loads(fixed)

            # Validate required keys are present
            required_keys = {"hair", "eyes", "clothing", "body", "distinguishing", "short_anchor_text"}
            missing = required_keys - set(anchor.keys())
            if missing:
                raise ValueError(f"Missing keys in anchor: {missing}")

            # Build derived fields programmatically
            full_anchor_text = (
                f"{anchor['hair']}, {anchor['eyes']}, {anchor['clothing']}, "
                f"{anchor['body']}, {anchor['distinguishing']}"
            )
            style_anchor = f"{style_trigger_word}, {style_name} style, vibrant colors, illustration"
            reference_prompt = (
                f"neutral standing pose, {full_anchor_text}, "
                f"facing forward, {style_trigger_word} style, illustration"
            )

            anchor["full_anchor_text"] = full_anchor_text
            anchor["short_anchor_text"] = anchor.get("short_anchor_text", full_anchor_text[:60])
            anchor["style_anchor"] = style_anchor
            anchor["reference_prompt"] = reference_prompt

            print(f"[Stage 2] Character anchor generated: {full_anchor_text[:80]}...")
            return anchor

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"[Stage 2] Anchor attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                prompt += (
                    "\n\nIMPORTANT: You must respond with ONLY valid JSON. "
                    "Do not use markdown code fences. Do not add any text before or after the JSON. "
                    "Keep all string values short and properly terminated."
                )
                time.sleep(1)
            else:
                raise RuntimeError(f"Failed to generate character anchor after {max_retries} attempts: {e}")

    raise RuntimeError("Unexpected exit from anchor retry loop")


def generate_storyboard(
    audio_data: AudioAnalysisResult,
    style_name: str,
    style_description: str,
    style_trigger_word: str = "",
    max_retries: int = 3,
    save_path: str = None,
    max_frames: int = None,
    generation_mode: str = "fast",
    style_key: str = "none",
    detected_mood: str = "neutral"
) -> Storyboard:
    """
    Main entry point: sends audio analysis to Groq and returns a
    structured Storyboard.

    v2 dual-mode: Two sequential calls:
      1. generate_character_anchor() — locked anchor
      2. Groq main call — full storyboard, mode-aware

    Parameters
    ----------
    audio_data : AudioAnalysisResult
        The complete audio fingerprint from Stage 1.
    style_name : str
        Human-readable style name (e.g. "MHA Anime").
    style_description : str
        Style description from the LoRA registry.
    style_trigger_word : str, optional
        Additional trigger words to append to the prompt for style guidance.
    max_retries : int
        Number of times to retry if Groq returns malformed JSON.
    save_path : str, optional
        If provided, saves the storyboard JSON to this path.
    max_frames : int, optional
        Target total frame count. When provided, the LLM will aim for this
        many frames (keyframes + continuations) distributed across scenes.
    generation_mode : str
        "fast" (default) or "quality" — selects the output schema and frame instructions.

    Returns
    -------
    Storyboard
        Structured visual plan with scenes and frame-by-frame directives.
    """
    from stages.stage1_audio import determine_scene_count

    num_scenes = determine_scene_count(audio_data, max_frames=max_frames)

    # ── Use unique moments detected in Stage 1 ──────────────────────────────
    # detect_unique_moments was already called in analyze_audio() with the same
    # max_frames, so audio_data.unique_moments should be populated. Fall back
    # to a fresh detection only if Stage 1 didn't compute them (e.g. cached
    # audio_data.json from an older run).
    unique_moments = audio_data.unique_moments if audio_data.unique_moments else []
    if not unique_moments:
        from stages.stage1_audio import detect_unique_moments
        unique_moments = detect_unique_moments(audio_data, num_scenes=num_scenes)
    if unique_moments:
        print(f"[Stage 2] Using {len(unique_moments)} unique moments from Stage 1 for keyframe placement")

    # ── CALL 1: Generate Character + Style Anchor ──────────────────────────
    # This is a small focused call (~200 tokens). It extracts a locked
    # character description from the audio mood + chosen style BEFORE
    # any scenes are designed. This anchor is then frozen for the entire video.
    anchor = generate_character_anchor(audio_data, style_name, style_trigger_word)

    # ── Task C: Guard against null / failed character anchor ────────────────
    if anchor is None or not anchor.get("short_anchor_text"):
        raise ValueError(
            "Character anchor generation failed — storyboard aborted. "
            "Check Groq API key and TPM limits."
        )

    # ── CALL 2: Generate Full Scene + Frame Plan ───────────────────────────
    from models.lora_registry import STYLE_LORAS, MOOD_GRADES

    style_info = STYLE_LORAS.get(style_key, STYLE_LORAS["none"])
    mood_info = MOOD_GRADES.get(detected_mood, MOOD_GRADES["neutral"])

    # Inject style and mood direction into system prompt
    style_block = f"\n\nART STYLE: {style_info['stage2_direction']}" if style_info.get("stage2_direction") else ""
    mood_block = (
        f"\nMOOD DIRECTION: {mood_info['stage2_direction']}"
        f"\nAVOID IN ALL SCENES: {mood_info['stage2_negative']}"
    ) if mood_info.get("stage2_direction") else ""

    enhanced_system_prompt = SYSTEM_PROMPT + style_block + mood_block

    # Extend the shared negative prompt with mood + style specifics
    extended_negative = mood_info.get("stage2_negative", "")
    style_neg = style_info.get("negative_add", "")
    if extended_negative or style_neg:
        # Prepend negatives to the anchor's style_anchor (already used in stage2 prompt) — they'll be
        # incorporated via build_storyboard_prompt's existing negative handling
        pass  # mood/style negatives are injected via system prompt blocks above

    prompt = build_storyboard_prompt(
        audio_data=audio_data,
        anchor=anchor,
        style_name=style_name,
        style_description=style_description,
        style_trigger_word=style_trigger_word,
        generation_mode=generation_mode,
        num_scenes=num_scenes,
        max_frames=max_frames,
        unique_moments=unique_moments,
    )

    print(f"[Stage 2] Generating storyboard for {num_scenes} scenes in {generation_mode} mode...")
    print(f"[Stage 2] Sending to Groq model: {STORYBOARD_MODEL}")

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=STORYBOARD_MODEL,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.75,
                max_tokens=5500  # Groq free-tier 8000 TPM limit (prompt ~2300 + 5500 = ~7800, safe margin)
            )

            raw_json = response.choices[0].message.content.strip()
            raw_json = strip_json_fences(raw_json)

            storyboard = parse_storyboard_json(raw_json)

            # Beat-align frame durations so transitions land on real beat hits
            from stages.beat_utils import compute_beat_aligned_frame_durations
            all_frames = []
            for scene in storyboard.scenes:
                all_frames.extend(scene.frames)
            if all_frames:
                durations = compute_beat_aligned_frame_durations(
                    beat_times=audio_data.beat_times,
                    frame_directives=all_frames,
                    total_duration=audio_data.duration,
                )
                for frame, dur in zip(all_frames, durations):
                    frame.hold_duration = dur

            print(f"[Stage 2] Storyboard parsed successfully.")
            print(f"[Stage 2] Main story: {storyboard.main_story[:100]}...")
            print(f"[Stage 2] Total scenes: {storyboard.total_scenes}")

            total_frames = sum(len(s.frames) for s in storyboard.scenes)
            kf_count = sum(1 for s in storyboard.scenes for f in s.frames if f.type.value == "KEYFRAME")
            cont_count = total_frames - kf_count
            print(f"[Stage 2] Frames: {total_frames} total ({kf_count} keyframes, {cont_count} continuations)")

            # ── Frame distribution enforcement ────────────────────────────────
            # Ensure each scene has at least 3 frames (1 keyframe + 2-3 continuations)
            # for a video-like feel (not a slideshow). If the LLM produced fewer,
            # duplicate continuation frames around keyframes to fill the target.
            if max_frames is not None and max_frames > 0:
                from stages.stage2_storyboard import _enforce_frame_distribution
                _enforce_frame_distribution(storyboard, max_frames, unique_moments, audio_data)

            # Warn if frame count exceeds the target
            if max_frames is not None and total_frames > max_frames:
                print(f"[Stage 2] WARNING: Generated {total_frames} frames, exceeding target of {max_frames}")

            # Inject anchor into parsed result for downstream use
            if generation_mode == "fast":
                storyboard.character_anchor = CharacterAnchor(**anchor)
                storyboard.style_anchor = anchor["style_anchor"]
            else:
                # Quality mode: inject the anchor's reference_prompt if not present
                if not storyboard.subject_reference_prompt:
                    storyboard.subject_reference_prompt = anchor.get("reference_prompt", "")
            # Inject character_anchor for both modes — used by Stage 3 for
            # {character_anchor} token resolution in keyframe prompts
            if not storyboard.character_anchor:
                storyboard.character_anchor = CharacterAnchor(**anchor)

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w") as f:
                    json.dump(storyboard.model_dump(), f, indent=2)
                print(f"[Stage 2] Storyboard saved to: {save_path}")

            return storyboard

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"[Stage 2] Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                # Retry with a stricter prompt to force JSON-only output
                prompt += (
                    "\n\nIMPORTANT: You must respond with ONLY valid JSON. "
                    "Do not use markdown code fences. Do not add any text before or after the JSON."
                )
                time.sleep(1)
            else:
                raise RuntimeError(f"Failed to parse storyboard JSON after {max_retries} attempts: {e}")

    raise RuntimeError("Unexpected exit from retry loop")


def parse_storyboard_json(raw_json: str) -> Storyboard:
    """
    Parses the JSON string from Groq into a Storyboard object.
    Handles common formatting issues.
    """
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        # Try fixing common JSON issues
        fixed = fix_json(raw_json)
        parsed = json.loads(fixed)

    # The character_anchor is always re-injected from the dedicated Call 1
    # anchor (see generate_storyboard), so strip any LLM-generated version to
    # avoid Pydantic validation failures when the LLM omits required fields.
    parsed.pop("character_anchor", None)
    storyboard = Storyboard(**parsed)
    return storyboard


def fix_json(text: str) -> str:
    """
    Attempts to fix common JSON formatting issues from LLM output.
    """
    text = text.strip()
    first_brace = text.find('{')
    if first_brace > 0:
        text = text[first_brace:]

    last_brace = text.rfind('}')
    if last_brace > 0 and last_brace < len(text) - 1:
        text = text[:last_brace + 1]

    # Fix trailing commas
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # Fix missing quotes around keys (common LLM mistake)
    text = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', text)

    return text


def strip_json_fences(text: str) -> str:
    """
    Strips markdown code fences from LLM response if present.
    Handles ```json, ```python, ```, etc.
    """
    text = re.sub(r'^```\w*\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    return text.strip()


def validate_storyboard(
    storyboard: Storyboard,
    generation_mode: str = None,
    max_frames: int = None
) -> list:
    """
    Validates the storyboard for common issues.
    Mode-aware: checks different things depending on generation_mode.

    Returns a list of warning messages (empty if all checks pass).
    """
    warnings = []
    mode = generation_mode or getattr(storyboard, 'generation_mode', None)
    if mode is None:
        mode = "quality"  # Default for backward compatibility

    if not storyboard.scenes:
        warnings.append("Storyboard has no scenes")
        return warnings

    for scene in storyboard.scenes:
        if not scene.frames:
            warnings.append(f"Scene {scene.scene_id} has no frames")
            continue

        # Check that first frame is a keyframe
        first_frame = scene.frames[0]
        if first_frame.type.value != "KEYFRAME":
            warnings.append(
                f"Scene {scene.scene_id} first frame is {first_frame.type}, expected KEYFRAME"
            )

        # Mode-specific validation
        if mode == "fast":
            # Fast mode: every frame should have full_prompt
            # CONT frames should NOT have modifier_prompt/strength
            if scene.scene_seed is None:
                warnings.append(f"Scene {scene.scene_id} has no scene_seed (required for Fast mode)")
            for frame in scene.frames:
                if not frame.full_prompt:
                    warnings.append(
                        f"Scene {scene.scene_id} frame {frame.frame_id}: no full_prompt"
                    )
                if frame.type.value == "CONTINUATION" and frame.modifier_prompt:
                    warnings.append(
                        f"Scene {scene.scene_id} frame {frame.frame_id}: CONTINUATION has modifier_prompt "
                        f"(should only have full_prompt in Fast mode)"
                    )
        else:
            # Quality mode: CONT frames should have modifier_prompt
            for frame in scene.frames:
                if frame.type.value == "CONTINUATION":
                    if not frame.modifier_prompt:
                        warnings.append(
                            f"Scene {scene.scene_id} frame {frame.frame_id}: CONTINUATION has no modifier_prompt"
                        )
                    if frame.strength is None:
                        warnings.append(
                            f"Scene {scene.scene_id} frame {frame.frame_id}: CONTINUATION has no strength"
                        )

        # Check frame timestamps are monotonic
        timestamps = [f.timestamp for f in scene.frames]
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1]:
                warnings.append(
                    f"Scene {scene.scene_id}: frame timestamps not monotonic "
                    f"at frame {scene.frames[i].frame_id}"
                )

        # Check scene end time
        last_ts = scene.frames[-1].timestamp
        if last_ts > scene.end_time:
            warnings.append(
                f"Scene {scene.scene_id}: last frame timestamp ({last_ts}) "
                f"exceeds end_time ({scene.end_time})"
            )

    return warnings


def _enforce_frame_distribution(storyboard, max_frames: int, unique_moments: list = None, audio_data = None):
    """
    Ensures each scene has enough continuation frames for a video-like feel.

    The LLM may not always produce 30 frames. If it produces fewer, we duplicate
    continuation frames around keyframes (within ±0.3s) to reach the target.
    This avoids slideshow pacing by ensuring each keyframe has 2-3 followers.

    Keyframes timestamps snap to nearest unique moment when available.
    """
    from models.schemas import FrameDirective, FrameType

    num_scenes = len(storyboard.scenes)
    if num_scenes == 0:
        return

    target_per_scene = max_frames // num_scenes
    if target_per_scene < 3:
        target_per_scene = 3  # minimum 3 frames per scene (1 KF + 2 CONT)

    total_frames = sum(len(s.frames) for s in storyboard.scenes)

    # Snap keyframe timestamps to nearest unique moment
    if unique_moments:
        um_times = [m["timestamp"] for m in unique_moments]
        for scene in storyboard.scenes:
            if scene.frames and scene.frames[0].type == FrameType.KEYFRAME:
                kf_ts = scene.frames[0].timestamp
                nearest_um = min(um_times, key=lambda u: abs(u - kf_ts))
                if abs(nearest_um - kf_ts) < 1.5:
                    scene.frames[0].timestamp = nearest_um

    # If we already have enough frames, just re-balance timestamps
    if total_frames >= max_frames:
        _rebalance_frame_timestamps(storyboard, audio_data)
        return

    # Add continuation frames to bring each scene up to target_per_scene.
    # The LLM may produce fewer frames than requested; this guarantees
    # each scene reaches the target so the final frame count matches max_frames.
    next_frame_id = max(f.frame_id for s in storyboard.scenes for f in s.frames) + 1

    for scene in storyboard.scenes:
        frames_needed = target_per_scene - len(scene.frames)
        if frames_needed <= 0:
            continue  # Scene already has enough frames

        # Duplicate the keyframe as the basis for continuation frames.
        kf = scene.frames[0]
        scene_end = scene.end_time

        # Determine keyframe timestamp (snap to unique moment if available)
        kf_ts = kf.timestamp

        # Spread continuation timestamps evenly between keyframe and scene end
        for i in range(1, frames_needed + 1):
            # Evenly distribute from just after keyframe to scene end
            fractional = i / (frames_needed + 1)
            ct = kf_ts + fractional * (scene_end - kf_ts)
            if ct > scene_end:
                ct = scene_end - 0.01 * i  # nudge back inside bounds

            # Base the continuation on the keyframe's content for consistency
            base_prompt = kf.full_prompt or kf.modifier_prompt or "character in scene, subtle motion"

            new_frame = FrameDirective(
                frame_id=next_frame_id,
                timestamp=round(ct, 2),
                type=FrameType.CONTINUATION,
                hold_duration=None,
                modifier_prompt=base_prompt,
                strength=0.35,
                what_changes="continuation of keyframe composition",
            )
            scene.frames.append(new_frame)
            next_frame_id += 1

    # Rebalance timestamps to fit within scene boundaries
    _rebalance_frame_timestamps(storyboard, audio_data)

    new_total = sum(len(s.frames) for s in storyboard.scenes)
    print(f"[Stage 2] Frame distribution enforced: {total_frames} → {new_total} frames (target: {max_frames})")


def _rebalance_frame_timestamps(storyboard, audio_data = None):
    """
    Ensures frame timestamps are monotonically increasing within each scene
    and span from scene.start_time to scene.end_time.
    """
    for scene in storyboard.scenes:
        n_frames = len(scene.frames)
        if n_frames == 0:
            continue
        if n_frames == 1:
            scene.frames[0].timestamp = scene.start_time
            continue

        # Evenly distribute timestamps from start_time to end_time,
        # but snap to beat-aligned times when possible
        if audio_data and audio_data.beat_times:
            for idx, frame in enumerate(scene.frames):
                ideal_t = scene.start_time + (scene.end_time - scene.start_time) * idx / (n_frames - 1)
                # Snap to nearest beat
                nearest_beat = min(audio_data.beat_times, key=lambda b: abs(b - ideal_t))
                if abs(nearest_beat - ideal_t) < 0.5:
                    frame.timestamp = round(nearest_beat, 2)
                else:
                    frame.timestamp = round(ideal_t, 2)
        else:
            for idx, frame in enumerate(scene.frames):
                if n_frames > 1:
                    frame.timestamp = round(
                        scene.start_time + (scene.end_time - scene.start_time) * idx / (n_frames - 1),
                        2
                    )
                else:
                    frame.timestamp = round(scene.start_time, 2)


if __name__ == "__main__":
    import argparse
    from stages.stage1_audio import analyze_audio
    from models.lora_registry import get_style, LORA_STYLES

    parser = argparse.ArgumentParser(description="Test Stage 2 storyboard generation")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("style_key", nargs="?", default="cyberpunk",
                        help="Style key (default: cyberpunk)")
    parser.add_argument("--mode", choices=["fast", "quality"], default="fast",
                        help="Generation mode (default: fast)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Target total frame count")
    args = parser.parse_args()

    style_key = args.style_key if args.style_key in LORA_STYLES else "cyberpunk"
    style = get_style(style_key)

    audio_file = args.audio_path
    analysis_path = f"./temp/{os.path.basename(audio_file)}_analysis.json"

    if os.path.exists(analysis_path):
        print(f"[Stage 2] Loading existing analysis: {analysis_path}")
        with open(analysis_path) as f:
            audio_data = AudioAnalysisResult(**json.load(f))
    else:
        print("[Stage 2] No cached analysis found, running Stage 1...")
        audio_data = analyze_audio(audio_file, save_path=analysis_path)

    storyboard = generate_storyboard(
        audio_data=audio_data,
        style_name=style["display_name"],
        style_description=style["description"],
        style_trigger_word=style["trigger_word"],
        save_path=f"./temp/{os.path.basename(audio_file)}_storyboard.json",
        generation_mode=args.mode,
        max_frames=args.max_frames
    )

    warnings = validate_storyboard(storyboard, generation_mode=args.mode, max_frames=args.max_frames)
    if warnings:
        print(f"\n[Stage 2] Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠ {w}")

    print(f"\n[Stage 2] Main story: {storyboard.main_story}")
    print(f"[Stage 2] Generation mode: {storyboard.generation_mode}")
    if storyboard.subject_description:
        print(f"[Stage 2] Subject: {storyboard.subject_description}")
    if storyboard.character_anchor:
        print(f"[Stage 2] Character anchor: {storyboard.character_anchor.full_anchor_text}")

    for scene in storyboard.scenes:
        seed_str = f", seed={scene.scene_seed}" if scene.scene_seed else ""
        print(f"\n  Scene {scene.scene_id}{seed_str}: {scene.start_time:.1f}-{scene.end_time:.1f}s")
        print(f"    Mood: {scene.mood}, Setting: {scene.setting}")
        for frame in scene.frames:
            t = "KEY" if frame.type.value == "KEYFRAME" else "CONT"
            print(f"    {t} frame {frame.frame_id} @ {frame.timestamp:.2f}s")
            if frame.full_prompt:
                print(f"      Prompt: {frame.full_prompt[:80]}...")
            elif frame.modifier_prompt:
                print(f"      Modifier: {frame.modifier_prompt[:80]}... (strength={frame.strength})")
