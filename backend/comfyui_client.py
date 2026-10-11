"""
ComfyUI Client — wraps the ComfyUI REST API for local image generation.

Exposes the same interface as HuggingFace's InferenceClient:
  text_to_image(prompt, model=None, negative_prompt=None, width, height,
                num_inference_steps, guidance_scale, seed, **kwargs)
  image_to_image(image, prompt, model=None, negative_prompt=None,
                 strength, width, height, num_inference_steps,
                 guidance_scale, seed, **kwargs)

Both methods return PIL.Image.Image, just like InferenceClient does.

Usage:
    from comfyui_client import ComfyUIClient
    client = ComfyUIClient()
    img = client.text_to_image(prompt="a cat", seed=42)
    img2 = client.image_to_image(image=img, prompt="a cat wearing a hat", strength=0.2)
"""

import os
import io
import time
import json
import requests
from PIL import Image

COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "http://127.0.0.1:8188")
COMFYUI_MODEL = os.environ.get("COMFYUI_MODEL", "meinmix_v12Final")
COMFYUI_TIMEOUT = int(os.environ.get("COMFYUI_TIMEOUT", "600"))  # 600s default for CPU/RX580
COMFYUI_MAX_RETRIES = int(os.environ.get("COMFYUI_MAX_RETRIES", "3"))


class ComfyUIClient:
    """Minimal REST client for ComfyUI's built-in HTTP server.

    The ComfyUI server exposes endpoints at /prompt, /history/<id>,
    /view, and /upload. This client submits workflow JSON graphs,
    polls for completion, and downloads output images as PIL Images.
    """

    def __init__(self, host: str = None, model_name: str = None):
        self.host = host or COMFYUI_HOST
        self.model_name = model_name or COMFYUI_MODEL
        self.session = requests.Session()
        # Verify API is reachable
        self._check_health()

    def _check_health(self):
        """Verify ComfyUI is running and responsive."""
        try:
            r = self.session.get(f"{self.host}/view", timeout=5)
            # /view without params returns 400 or empty, but the host is reachable
        except requests.ConnectionError:
            raise ConnectionError(f"ComfyUI not reachable at {self.host}")

    def submit_prompt(self, workflow: dict) -> str:
        """POST a workflow JSON to ComfyUI. Returns prompt_id."""
        resp = self.session.post(
            f"{self.host}/prompt",
            json={"prompt": workflow, "extra_data": {}}
        )
        resp.raise_for_status()
        return resp.json()["prompt_id"]

    def wait_for_result(self, prompt_id: str, timeout: int = None) -> dict:
        """
        Poll /history until the prompt completes. Returns the history entry.

        On timeout, raises TimeoutError with the prompt_id for debugging.
        The default timeout is COMFYUI_TIMEOUT (600s) to accommodate CPU
        generation on GPUs like the RX580 that lack CUDA support.
        """
        if timeout is None:
            timeout = COMFYUI_TIMEOUT
        start = time.time()
        while time.time() - start < timeout:
            resp = self.session.get(f"{self.host}/history/{prompt_id}")
            if resp.status_code == 404:
                time.sleep(0.5)
                continue
            history = resp.json()
            if prompt_id in history:
                entry = history[prompt_id]
                if entry.get("status", {}).get("completed"):
                    return entry
            time.sleep(0.5)
        raise TimeoutError(
            f"ComfyUI prompt {prompt_id} did not complete within {timeout}s. "
            f"This may indicate VRAM/OOM issues or slow CPU generation. "
            f"Check the ComfyUI server logs for details."
        )

    def get_image(self, prompt_id: str, filename: str,
                  subfolder: str = "", img_type: str = "output") -> Image.Image:
        """Download an output image via /view endpoint. Returns PIL Image."""
        resp = self.session.get(
            f"{self.host}/view",
            params={"prompt_id": prompt_id, "filename": filename,
                    "subfolder": subfolder, "type": img_type}
        )
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def upload_image(self, image: Image.Image, filename: str) -> str:
        """Upload a PIL image to ComfyUI's input/ directory. Returns the filename."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        resp = self.session.post(
            f"{self.host}/upload",
            files={"image": (filename, buf, "image/png")},
            data={"dest": "input"}
        )
        resp.raise_for_status()
        return resp.json().get("name", filename)

    def text_to_image(self, prompt, model=None, negative_prompt=None,
                      width=768, height=768, num_inference_steps=20,
                      guidance_scale=7.5, seed=None, **kwargs) -> Image.Image:
        """
        Mirrors InferenceClient.text_to_image() interface.
        Generates an image from a text prompt using ComfyUI.

        Retries on failure (timeout, OOM, etc.) up to COMFYUI_MAX_RETRIES times.
        """
        last_error = None
        for attempt in range(1, COMFYUI_MAX_RETRIES + 1):
            try:
                workflow = build_t2i_workflow(
                    ckpt_name=self.model_name,
                    positive_prompt=prompt,
                    negative_prompt=negative_prompt or "",
                    width=width, height=height,
                    steps=num_inference_steps,
                    cfg=guidance_scale,
                    seed=seed or 0,
                    filename_prefix=f"audiostory_{int(time.time())}_{attempt}"
                )
                prompt_id = self.submit_prompt(workflow)
                history = self.wait_for_result(prompt_id)
                # Extract the first output image from SaveImage node (node "7")
                # ComfyUI 0.3.48 returns images as list of dicts: [{"filename":..., "subfolder":..., "type":...}]
                images = history["outputs"]["7"]["images"]
                img_info = images[0]
                filename = img_info["filename"]
                subfolder = img_info.get("subfolder", "")
                img_type = img_info.get("type", "output")
                return self.get_image(prompt_id, filename, subfolder, img_type)
            except (TimeoutError, requests.RequestException, KeyError, IndexError) as e:
                last_error = e
                if attempt < COMFYUI_MAX_RETRIES:
                    print(f"[ComfyUI] text_to_image attempt {attempt}/{COMFYUI_MAX_RETRIES} failed: {e}")
                    print(f"[ComfyUI] Retrying in 5s...")
                    time.sleep(5)
        raise RuntimeError(
            f"ComfyUI text_to_image failed after {COMFYUI_MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    def image_to_image(self, image, prompt, model=None, negative_prompt=None,
                       strength=0.3, width=768, height=768,
                       num_inference_steps=20, guidance_scale=7.5,
                       seed=None, **kwargs) -> Image.Image:
        """
        Mirrors InferenceClient.image_to_image() interface.
        Generates an image by modifying a reference image using ComfyUI img2img.
        strength=0.82 -> denoise=0.82 (heavy change), strength=0.22 -> denoise=0.22 (subtle).

        Retries on failure up to COMFYUI_MAX_RETRIES times.
        """
        last_error = None
        for attempt in range(1, COMFYUI_MAX_RETRIES + 1):
            try:
                # Upload the init image
                timestamp = int(time.time())
                fname = f"audiostory_img2img_{timestamp}_{attempt}.png"
                self.upload_image(image, fname)

                workflow = build_img2img_workflow(
                    ckpt_name=self.model_name,
                    positive_prompt=prompt,
                    negative_prompt=negative_prompt or "",
                    init_image_filename=fname,
                    denoise=strength,  # img2img strength maps directly to KSampler.denoise
                    width=width, height=height,
                    steps=num_inference_steps,
                    cfg=guidance_scale,
                    seed=seed or 0,
                    filename_prefix=f"audiostory_img2img_{timestamp}_{attempt}"
                )
                prompt_id = self.submit_prompt(workflow)
                history = self.wait_for_result(prompt_id)
                # Extract the first output image from SaveImage node (node "8")
                # ComfyUI 0.3.48 returns images as list of dicts: [{"filename":..., "subfolder":..., "type":...}]
                images = history["outputs"]["8"]["images"]
                img_info = images[0]
                filename = img_info["filename"]
                subfolder = img_info.get("subfolder", "")
                img_type = img_info.get("type", "output")
                return self.get_image(prompt_id, filename, subfolder, img_type)
            except (TimeoutError, requests.RequestException, KeyError, IndexError) as e:
                last_error = e
                if attempt < COMFYUI_MAX_RETRIES:
                    print(f"[ComfyUI] image_to_image attempt {attempt}/{COMFYUI_MAX_RETRIES} failed: {e}")
                    print(f"[ComfyUI] Retrying in 5s...")
                    time.sleep(5)
        raise RuntimeError(
            f"ComfyUI image_to_image failed after {COMFYUI_MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Workflow builders — construct ComfyUI node-graph JSON
# ─────────────────────────────────────────────────────────────────────────

def build_t2i_workflow(ckpt_name, positive_prompt, negative_prompt,
                       width, height, steps, cfg, seed, filename_prefix) -> dict:
    """Builds a ComfyUI text-to-image workflow dict."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": positive_prompt}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": negative_prompt}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "seed": seed, "steps": steps,
                         "cfg": cfg, "sampler_name": "euler", "scheduler": "normal",
                         "denoise": 1.0,
                         "positive": ["2", 0], "negative": ["3", 0],
                         "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode",
              "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"images": ["6", 0], "filename_prefix": filename_prefix}}
    }


def build_img2img_workflow(ckpt_name, positive_prompt, negative_prompt,
                           init_image_filename, denoise, width, height,
                           steps, cfg, seed, filename_prefix) -> dict:
    """Builds a ComfyUI image-to-image workflow dict."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": positive_prompt}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": negative_prompt}},
        "4": {"class_type": "LoadImage",
              "inputs": {"image": init_image_filename}},
        "5": {"class_type": "VAEEncode",
              "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}},
        "6": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "seed": seed, "steps": steps,
                         "cfg": cfg, "sampler_name": "euler", "scheduler": "normal",
                         "denoise": denoise,
                         "positive": ["2", 0], "negative": ["3", 0],
                         "latent_image": ["5", 0]}},
        "7": {"class_type": "VAEDecode",
              "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0], "filename_prefix": filename_prefix}}
    }
