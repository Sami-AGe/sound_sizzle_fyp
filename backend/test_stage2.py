"""
Quick test script for Stage 2 - storyboard generation.

Supports both Fast and Quality generation modes via --mode flag or
GENERATION_MODE env var.

Run:
  cd backend && python3 test_stage2.py --mode fast
  cd backend && GENERATION_MODE=quality python3 test_stage2.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join("..", ".env"))

from models.schemas import AudioAnalysisResult
from stages.stage2_storyboard import generate_storyboard, validate_storyboard
from models.lora_registry import get_style, LORA_STYLES

# ── Parse args ───────────────────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser(description="Test Stage 2 storyboard generation")
parser.add_argument("--audio", default="../temp/85258bf0/audio_data.json",
                    help="Path to audio_data.json (default: existing analysis)")
parser.add_argument("--style", default="cyberpunk",
                    help="Style key (default: cyberpunk)")
parser.add_argument("--mode", choices=["fast", "quality"], default=None,
                    help="Generation mode (default: GENERATION_MODE env or fast)")
parser.add_argument("--max-frames", type=int, default=None,
                    help="Target total frame count")
parser.add_argument("--save", default=None,
                    help="Custom save path for storyboard JSON")
args = parser.parse_args()

generation_mode = args.mode or os.environ.get("GENERATION_MODE", "fast")
max_frames = args.max_frames or int(os.environ.get("MAX_FRAMES", "15")) or None

# ── Load audio data ───────────────────────────────────────────────────────────
audio_path = args.audio
if not os.path.exists(audio_path):
    print(f"ERROR: No audio analysis at {audio_path}")
    print("Run Stage 1 first to generate audio_data.json")
    sys.exit(1)

with open(audio_path) as f:
    audio_data = AudioAnalysisResult(**json.load(f))

print(f"Audio loaded: {audio_data.duration}s, mood={audio_data.detected_mood}")
print(f"Transcript: {audio_data.transcript}")
print(f"Beats: {len(audio_data.beat_times)}, Onsets: {len(audio_data.onset_times)}")
print(f"Energy sections: {len(audio_data.energy_sections)}")
for s in audio_data.energy_sections:
    print(f"  {s['label']}: {s['start']:.1f}-{s['end']:.1f}s, energy={s['avg_energy']:.2f}, {s['energy_label']}")

style_key = args.style if args.style in LORA_STYLES else "cyberpunk"
style = get_style(style_key)
print(f"\nStyle: {style['display_name']}")
print(f"Mode: {generation_mode}")
print(f"Max frames: {max_frames}")

# ── Generate storyboard ────────────────────────────────────────────────────────
save_path = args.save or f"../temp/test_{generation_mode}_storyboard.json"

storyboard = generate_storyboard(
    audio_data=audio_data,
    style_name=style["display_name"],
    style_description=style["description"],
    style_trigger_word=style["trigger_word"],
    save_path=save_path,
    generation_mode=generation_mode,
    max_frames=max_frames
)

# ── Validate ──────────────────────────────────────────────────────────────────
warnings = validate_storyboard(storyboard, generation_mode=generation_mode, max_frames=max_frames)
if warnings:
    print(f"\n⚠️  Validation warnings ({len(warnings)}):")
    for w in warnings:
        print(f"  - {w}")

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"STORYBOARD SUMMARY")
print(f"{'='*60}")
print(f"Main story: {storyboard.main_story}")
print(f"Generation mode: {storyboard.generation_mode}")
print(f"Total scenes: {storyboard.total_scenes}")

if storyboard.subject_description:
    print(f"Subject: {storyboard.subject_description}")

if storyboard.character_anchor:
    print(f"\nCharacter anchor: {storyboard.character_anchor.full_anchor_text}")
    print(f"Style anchor: {storyboard.style_anchor}")
    word_count = len(storyboard.character_anchor.full_anchor_text.split())
    print(f"Anchor word count: {word_count} (target < 60)")
    if word_count > 60:
        print(f"  ⚠️  WARNING: exceeds 60 words!")

if storyboard.emotional_arc:
    arc = storyboard.emotional_arc
    print(f"\nEmotional arc:")
    print(f"  Intro:    {arc.intro}")
    print(f"  Buildup:  {arc.buildup}")
    print(f"  Climax:   {arc.climax}")
    print(f"  Outro:    {arc.outro}")

total_frames = sum(len(s.frames) for s in storyboard.scenes)
kf_count = sum(1 for s in storyboard.scenes for f in s.frames if f.type.value == "KEYFRAME")
print(f"\nFrames: {total_frames} total ({kf_count} keyframes, {total_frames - kf_count} continuations)")

for scene in storyboard.scenes:
    seed_str = f", seed={scene.scene_seed}" if scene.scene_seed else ""
    print(f"\n  Scene {scene.scene_id}{seed_str}: {scene.start_time:.1f}-{scene.end_time:.1f}s")
    print(f"    Mood: {scene.mood}, Setting: {scene.setting}")
    print(f"    Color palette: {scene.color_palette}")
    print(f"    Transition: {scene.transition_from_previous}")
    for frame in scene.frames:
        t = "KEY " if frame.type.value == "KEYFRAME" else "CONT"
        print(f"    {t} frame {frame.frame_id} @ {frame.timestamp:.2f}s")
        if frame.full_prompt:
            print(f"      Prompt: {frame.full_prompt[:80]}...")
        elif frame.modifier_prompt:
            print(f"      Modifier: {frame.modifier_prompt[:80]}... (strength={frame.strength})")
        if frame.what_changes:
            print(f"      Changes: {frame.what_changes}")

print(f"\n{'='*60}")
if total_frames > (max_frames or 999):
    print(f"⚠️  Frame count ({total_frames}) exceeds target ({max_frames})")
else:
    print(f"✅ Storyboard saved to: {save_path}")
