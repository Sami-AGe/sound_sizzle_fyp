"""
Controlled test: Verify a complete Stage 3 model stack.

Two-model approach:
  - stabilityai/stable-diffusion-3-medium via replicate: text_to_image (reference)
  - black-forest-labs/FLUX.1-Kontext-dev via replicate: image_to_image (keyframes + continuations)

Already verified: SD3M text_to_image works.
Now testing: Kontext img2img (using SD3M output as input).
"""
import os
import sys
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join("..", ".env"))

from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Working models
TEXT_TO_IMAGE_MODEL = "stabilityai/stable-diffusion-3-medium"
IMAGE_TO_IMAGE_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
PROVIDER = "replicate"

TEST_PROMPT = "A cinematic photograph of a lone person standing on a mountain at sunset, cyberpunk style, 8k"
MODIFY_PROMPT = "A cyberpunk hacker in a neon-lit alley, rain falling, dramatic lighting, 8k"

OUTPUT_DIR = "../temp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create client with replicate provider
client = InferenceClient(
    token=HF_TOKEN,
    provider=PROVIDER,
    timeout=180,
)


def verify_image(image, label):
    arr = np.array(image)
    std = np.std(arr.reshape(-1, arr.shape[-1]), axis=0) if arr.ndim == 3 else [1]
    mean_std = float(np.mean(std))
    print(f"  [{label}] size={image.size}, mode={image.mode}, color_std={mean_std:.2f}")
    return mean_std > 10 and image.size[0] > 100


print("="*50)
print("Stage 3 Controlled Test — Two-Model Stack")
print("="*50)

# Step 1: Generate reference via text_to_image (SD3M via replicate)
print(f"\n[1] text_to_image: {TEXT_TO_IMAGE_MODEL} via {PROVIDER}")
ref_image = client.text_to_image(
    prompt=TEST_PROMPT,
    model=TEXT_TO_IMAGE_MODEL,
    negative_prompt="bad anatomy, deformed, blurry, low quality, watermark, text, photorealistic, photo, realistic",
    num_inference_steps=20,
    guidance_scale=7.5,
    width=768,
    height=768,
)
if not verify_image(ref_image, "reference"):
    print("  ❌ Reference image failed verification")
    sys.exit(1)
ref_path = os.path.join(OUTPUT_DIR, "test_reference.png")
ref_image.save(ref_path)
print(f"  Saved: {ref_path} ({os.path.getsize(ref_path)} bytes)")

# Step 2: Modify via image_to_image (Kontext via replicate) with strength=0.20
print(f"\n[2] image_to_image: {IMAGE_TO_IMAGE_MODEL} via {PROVIDER}, strength=0.20")
cont_image = client.image_to_image(
    image=ref_image,
    prompt=MODIFY_PROMPT,
    model=IMAGE_TO_IMAGE_MODEL,
    negative_prompt="bad anatomy, deformed, blurry, low quality, watermark, text, photorealistic, photo, realistic",
    num_inference_steps=20,
    guidance_scale=7.5,
    strength=0.20,  # passed via **kwargs
)
if not verify_image(cont_image, "continuation"):
    print("  ❌ Continuation image failed verification")
    sys.exit(1)
cont_path = os.path.join(OUTPUT_DIR, "test_continuation.png")
cont_image.save(cont_path)
print(f"  Saved: {cont_path} ({os.path.getsize(cont_path)} bytes)")

# Step 3: Keyframe via image_to_image with strength=0.82
print(f"\n[3] image_to_image (keyframe): {IMAGE_TO_IMAGE_MODEL} via {PROVIDER}, strength=0.82")
kf_image = client.image_to_image(
    image=ref_image,
    prompt=TEST_PROMPT + ", on a rooftop overlooking neon city",
    model=IMAGE_TO_IMAGE_MODEL,
    negative_prompt="bad anatomy, deformed, blurry, low quality, watermark, text, photorealistic, photo, realistic",
    num_inference_steps=20,
    guidance_scale=7.5,
    strength=0.82,
)
if not verify_image(kf_image, "keyframe"):
    print("  ❌ Keyframe image failed verification")
    sys.exit(1)
kf_path = os.path.join(OUTPUT_DIR, "test_keyframe.png")
kf_image.save(kf_path)
print(f"  Saved: {kf_path} ({os.path.getsize(kf_path)} bytes)")

print(f"\n✅ ALL TESTS PASSED")
print(f"  text_to_image model: {TEXT_TO_IMAGE_MODEL}")
print(f"  image_to_image model: {IMAGE_TO_IMAGE_MODEL}")
print(f"  provider: {PROVIDER}")
print(f"\nReady to update stage3_images.py with this configuration.")
