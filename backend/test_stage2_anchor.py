"""
Test script: Stage 2 Call 1 — Character Anchor Generation

Tests Groq Call 1 in isolation — anchor generation without the full storyboard.
No HuggingFace credits needed (Groq only).

Run: cd backend && python3 test_stage2_anchor.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join("..", ".env"))

from stages.stage2_storyboard import generate_character_anchor
from models.schemas import CharacterAnchor, AudioAnalysisResult

# Load existing Stage 1 output
audio_path = "../temp/85258bf0/audio_data.json"
if not os.path.exists(audio_path):
    print(f"ERROR: No audio analysis at {audio_path}")
    print("Run Stage 1 first.")
    sys.exit(1)

with open(audio_path) as f:
    audio_data = AudioAnalysisResult(**json.load(f))

print(f"Audio loaded: {audio_data.duration}s, mood={audio_data.detected_mood}")
print(f"Transcript: {audio_data.transcript}")
print()

# Test with different styles
styles = [
    ("mha_anime", "MHA Anime", "mha style, anime, shonen, vibrant colors, hero costume, dynamic pose"),
    ("cyberpunk", "Cyberpunk Neon", "cyberpunk style, neon lights, dark city, futuristic, holograms, rain"),
    ("lofi_pastel", "Lofi Pastel", "lofi style, soft pastel, warm illustration, gentle, cozy, dreamy"),
]

for style_key, style_name, trigger_word in styles:
    print(f"\n{'='*60}")
    print(f"Style: {style_name}")
    print(f"{'='*60}")

    anchor = generate_character_anchor(audio_data, style_name, trigger_word)

    # Print the raw JSON
    print(json.dumps(anchor, indent=2))

    # Validate
    try:
        validated = CharacterAnchor(**anchor)
        print(f"\n✅ Valid CharacterAnchor model")
        print(f"   full_anchor_text: {validated.full_anchor_text}")
        print(f"   style_anchor: {validated.style_anchor}")
        print(f"   reference_prompt: {validated.reference_prompt[:80]}...")

        # Check word count of full_anchor_text
        word_count = len(validated.full_anchor_text.split())
        print(f"   Word count: {word_count} (must be under 60)")
        if word_count > 60:
            print(f"   ⚠️  WARNING: full_anchor_text exceeds 60 words!")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)

print(f"\n{'='*60}")
print(f"✅ All character anchor tests passed!")
