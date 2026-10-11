"""
Test script: Stage 2 + Stage 3 in FAST mode

End-to-end test: Generates a storyboard (two-call Groq) in Fast mode,
then generates all frames via text_to_image (SD3-Medium) with scene seed locking.
All frames use the Character Anchor tokens for consistency.

Requires: Groq API key + HuggingFace credits (text_to_image via SD3-Medium).

Run: cd backend && GENERATION_MODE=fast MAX_FRAMES=12 python3 test_stage3_fast.py
"""
import json, os, sys, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join("..", ".env"))

from models.schemas import AudioAnalysisResult
from models.lora_registry import get_style
from stages.stage2_storyboard import generate_storyboard, validate_storyboard
from stages.stage3_images import generate_all_frames, save_frameset_metadata

# ── Load existing Stage 1 output ─────────────────────────────────────────────
audio_path = "../temp/85258bf0/audio_data.json"
with open(audio_path) as f:
    audio_data = AudioAnalysisResult(**json.load(f))

print(f"Audio loaded: {audio_data.duration}s, mood={audio_data.detected_mood}")

style = get_style("cyberpunk")
print(f"Style: {style['display_name']}")

generation_mode = os.environ.get("GENERATION_MODE", "fast")
max_frames = int(os.environ.get("MAX_FRAMES", "12"))
print(f"Generation mode: {generation_mode}")
print(f"Max frames target: {max_frames}")

# ── Stage 2: Generate storyboard (Fast mode) ──────────────────────────────────
print(f"\n{'='*60}")
print(f"STAGE 2: Storyboard (Fast mode)")
print(f"{'='*60}")

storyboard = generate_storyboard(
    audio_data=audio_data,
    style_name=style["display_name"],
    style_description=style["description"],
    style_trigger_word=style["trigger_word"],
    generation_mode=generation_mode,
    max_frames=max_frames,
    save_path=f"../temp/test_fast_storyboard.json"
)

warnings = validate_storyboard(storyboard, generation_mode=generation_mode, max_frames=max_frames)
if warnings:
    print(f"\n⚠️  Validation warnings ({len(warnings)}):")
    for w in warnings:
        print(f"  - {w}")

print(f"\nStoryboard generated!")
print(f"  Mode: {storyboard.generation_mode}")
print(f"  Scenes: {storyboard.total_scenes}")
if storyboard.character_anchor:
    print(f"  Character anchor: {storyboard.character_anchor.full_anchor_text[:80]}...")
if storyboard.style_anchor:
    print(f"  Style anchor: {storyboard.style_anchor}")

total_frames = sum(len(s.frames) for s in storyboard.scenes)
kf_count = sum(1 for s in storyboard.scenes for f in s.frames if f.type.value == "KEYFRAME")
print(f"  Frames: {total_frames} ({kf_count} keyframes, {total_frames - kf_count} continuations)")

# Verify Fast mode structure
for scene in storyboard.scenes:
    assert scene.scene_seed is not None, f"Scene {scene.scene_id} missing scene_seed!"
    for frame in scene.frames:
        assert frame.full_prompt is not None, f"Frame {frame.frame_id} missing full_prompt!"
        if frame.type.value == "CONTINUATION":
            assert frame.modifier_prompt is None, f"Frame {frame.frame_id} should NOT have modifier_prompt in Fast mode!"
            assert "{character_anchor}" in frame.full_prompt, f"Frame {frame.frame_id} full_prompt missing {{character_anchor}} token!"

print(f"\n✅ Fast mode storyboard structure validated!")

# ── Stage 3: Generate all frames (Fast mode) ──────────────────────────────────
print(f"\n{'='*60}")
print(f"STAGE 3: Image Generation (Fast mode)")
print(f"{'='*60}")

job_id = f"fast_{uuid.uuid4().hex[:8]}"
print(f"Job ID: {job_id}")

# Allow override of generation steps for faster testing
test_steps = int(os.environ.get("TEST_STEPS", "20"))
gen_params = {"steps": test_steps} if test_steps != 20 else None
if gen_params:
    print(f"Using reduced steps: {test_steps}")

frame_set = generate_all_frames(
    storyboard=storyboard,
    style_name=style["display_name"],
    style_trigger_word=style["trigger_word"],
    job_id=job_id,
    output_dir="../temp",
    gen_params=gen_params,
)

save_frameset_metadata(frame_set, f"../temp/{job_id}_frameset.json")

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"  Total frames: {frame_set.total_frames}")
print(f"  Generation mode: {frame_set.generation_mode}")
print(f"  Reference image: {frame_set.reference_image_path}")

# Verify each frame image exists and is non-trivial
print(f"\n  Frame verification:")
all_ok = True
for f in frame_set.frames:
    exists = os.path.exists(f.image_path)
    size = os.path.getsize(f.image_path) if exists else 0
    status = "✅" if exists and size > 1000 else "❌"
    if not (exists and size > 1000):
        all_ok = False
    print(f"    {status} Frame {f.frame_id} ({f.type}): {f.image_path} ({size:,} bytes)")

if all_ok:
    print(f"\n✅ All {frame_set.total_frames} images generated and verified!")
    print(f"   Output directory: ../temp/{job_id}/frames/")
    print(f"   Frameset metadata: ../temp/{job_id}_frameset.json")
else:
    print(f"\n❌ Some images missing or empty")
    sys.exit(1)
