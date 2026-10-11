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
    full_anchor_text, style_anchor, reference_prompt.

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
  "distinguishing": "one unique visual trait (energy aura, weapon, accessory, etc.)"
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
            required_keys = {"hair", "eyes", "clothing", "body", "distinguishing"}
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
    generation_mode: str = "fast"
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

    # ── CALL 1: Generate Character + Style Anchor ──────────────────────────
    # This is a small focused call (~200 tokens). It extracts a locked
    # character description from the audio mood + chosen style BEFORE
    # any scenes are designed. This anchor is then frozen for the entire video.
    anchor = generate_character_anchor(audio_data, style_name, style_trigger_word)

    # ── CALL 2: Generate Full Scene + Frame Plan ───────────────────────────
    prompt = build_storyboard_prompt(
        audio_data=audio_data,
        anchor=anchor,
        style_name=style_name,
        style_description=style_description,
        style_trigger_word=style_trigger_word,
        generation_mode=generation_mode,
        num_scenes=num_scenes,
        max_frames=max_frames
    )

    print(f"[Stage 2] Generating storyboard for {num_scenes} scenes in {generation_mode} mode...")
    print(f"[Stage 2] Sending to Groq model: {STORYBOARD_MODEL}")

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=STORYBOARD_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.75,
                max_tokens=8192
            )

            raw_json = response.choices[0].message.content.strip()
            raw_json = strip_json_fences(raw_json)

            storyboard = parse_storyboard_json(raw_json)

            print(f"[Stage 2] Storyboard parsed successfully.")
            print(f"[Stage 2] Main story: {storyboard.main_story[:100]}...")
            print(f"[Stage 2] Total scenes: {storyboard.total_scenes}")

            total_frames = sum(len(s.frames) for s in storyboard.scenes)
            kf_count = sum(1 for s in storyboard.scenes for f in s.frames if f.type.value == "KEYFRAME")
            cont_count = total_frames - kf_count
            print(f"[Stage 2] Frames: {total_frames} total ({kf_count} keyframes, {cont_count} continuations)")

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
