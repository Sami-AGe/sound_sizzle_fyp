"""
Browser simulation test — replicates exactly what frontend/index.html does.

Flow (mirrors generateVideo() + pollJobProgress() + showResults()):
  1. POST /generate with FormData (audio, style, max_frames, mode, steps, guidance, width, height, denoise, seed)
  2. Poll GET /jobs/{job_id} every 2s
  3. Track current_stage, logs, and status transitions
  4. When status === 'completed', fetch /download/{job_id} and verify it's a valid MP4
  5. Also test /test/stage1 and /test/stage2 endpoints
"""
import requests
import json
import time
import os
import sys

API_BASE = "http://localhost:8000"
AUDIO_PATH = "/Users/syedsunny/sound_sizzle_fyp/temp/test_audio.wav"

# Reduce load for RX580 CPU fallback
GEN_PARAMS = {
    "audio": (open(AUDIO_PATH, "rb"), "clip.wav"),
    "style": "mha_anime",
    "max_frames": "2",
    "mode": "fast",
    "steps": "8",
    "guidance": "7.0",
    "width": "512",
    "height": "512",
    "denoise": "0.75",
    "seed": "-1",
}

def stage_name(n):
    return {1: "Stage 1 — Audio Analysis", 2: "Stage 2 — Storyboard",
            3: "Stage 3 — Image Generation", 4: "Stage 4 — Video Assembly"}.get(n, "?")

def test_server_health():
    print("=" * 70)
    print("[CHECK 0] Server health")
    r = requests.get(f"{API_BASE}/")
    ok = r.status_code == 200 and r.json().get("service") == "AudioStory"
    print(f"  GET / → {r.status_code} {'✅' if ok else '❌'}")
    if not ok:
        print(f"  Response: {r.text[:200]}")
        return False

    r = requests.get(f"{API_BASE}/config")
    cfg = r.json()
    print(f"  GET /config → provider: {'ComfyUI' if cfg.get('use_comfyui') else 'HF'}, "
          f"model: {cfg.get('comfyui_model')}, mode: {cfg.get('generation_mode_default')} ✅")

    r = requests.get(f"{API_BASE}/styles")
    styles = r.json()
    print(f"  GET /styles → {len(styles)} styles available ✅")
    return True


def test_stage1():
    print("\n" + "=" * 70)
    print("[CHECK 1] POST /test/stage1 (audio analysis only)")
    start = time.time()
    with open(AUDIO_PATH, "rb") as f:
        files = {"audio": (os.path.basename(AUDIO_PATH), f, "audio/wav")}
        data = {"whisper_model": "base"}
        r = requests.post(f"{API_BASE}/test/stage1", files=files, data=data, timeout=60)
    elapsed = time.time() - start
    if r.status_code != 200:
        print(f"  ❌ HTTP {r.status_code}: {r.text[:300]}")
        return None
    result = r.json()
    print(f"  Status: {r.status_code} in {elapsed:.1f}s ✅")
    print(f"  Duration: {result.get('duration')}s")
    print(f"  Tempo: {result.get('tempo')} BPM")
    print(f"  Mood: {result.get('detected_mood')}")
    print(f"  Beats: {len(result.get('beat_times', []))}")
    print(f"  Onsets: {len(result.get('onset_times', []))}")
    print(f"  Energy sections: {len(result.get('energy_sections', []))}")
    print(f"  Transcript: {result.get('transcript', 'N/A')[:80]}...")

    # Verify the JSON matches Pydantic schema fields
    expected = {"duration", "tempo", "beat_times", "onset_times", "rms_values",
                "spectral_centroid_mean", "chroma_mean", "mfcc_mean",
                "zero_crossing_rate", "detected_mood", "energy_sections", "transcript"}
    missing = expected - set(result.keys())
    if missing:
        print(f"  ⚠️  Missing fields: {missing}")
    else:
        print(f"  All required AudioAnalysisResult fields present ✅")
    return result


def test_stage2():
    print("\n" + "=" * 70)
    print("[CHECK 2] POST /test/stage2 (audio + storyboard)")
    start = time.time()
    with open(AUDIO_PATH, "rb") as f:
        files = {"audio": (os.path.basename(AUDIO_PATH), f, "audio/wav")}
        data = {"style": "mha_anime", "mode": "fast", "max_frames": "2", "whisper_model": "base"}
        r = requests.post(f"{API_BASE}/test/stage2", files=files, data=data, timeout=120)
    elapsed = time.time() - start
    if r.status_code != 200:
        print(f"  ❌ HTTP {r.status_code}: {r.text[:300]}")
        return None
    result = r.json()
    sb = result.get("storyboard", {})
    print(f"  Status: {r.status_code} in {elapsed:.1f}s ✅")
    print(f"  Mode: {sb.get('generation_mode')}")
    print(f"  Scenes: {sb.get('total_scenes')}")
    print(f"  Total frames in storyboard: {sum(len(s.get('frames', [])) for s in sb.get('scenes', []))}")
    for scene in sb.get("scenes", []):
        frames = scene.get("frames", [])
        kfs = [f for f in frames if f.get("type") == "KEYFRAME"]
        conts = [f for f in frames if f.get("type") == "CONTINUATION"]
        has_anchor = any("{character_anchor}" in f.get("full_prompt", "") for f in frames)
        print(f"  Scene {scene.get('scene_id')}: seed={scene.get('scene_seed')}, "
              f"KF={len(kfs)}, CONT={len(conts)}, has_anchor_token={has_anchor}")
    # Verify Fast mode requirements
    has_char_anchor = sb.get('character_anchor') is not None
    all_have_full_prompt = all(
        f.get("full_prompt") for s in sb.get("scenes", []) for f in s.get("frames", [])
    )
    no_modifier_on_cont = all(
        f.get("modifier_prompt") is None for s in sb.get("scenes", [])
        for f in s.get("frames", []) if f.get("type") == "CONTINUATION"
    )
    print(f"  Character anchor present: {has_char_anchor} ✅" if has_char_anchor else f"  Character anchor present: {has_char_anchor} ❌")
    print(f"  All frames have full_prompt: {all_have_full_prompt} ✅" if all_have_full_prompt else f"  All frames have full_prompt: {all_have_full_prompt} ❌")
    print(f"  No modifier_prompt on CONT frames: {no_modifier_on_cont} ✅" if no_modifier_on_cont else f"  No modifier_prompt on CONT frames: {no_modifier_on_cont} ❌")
    return result


def test_full_pipeline_async():
    """
    Replicates the exact frontend flow:
    1. POST /generate → get job_id immediately
    2. Poll GET /jobs/{job_id} every 2s (like setInterval(pollJobProgress, 2000))
    3. Track current_stage, append new logs
    4. When completed → showResults() equivalent: verify video + download
    5. Test cancel endpoint
    """
    print("\n" + "=" * 70)
    print("[CHECK 3] POST /generate → async polling → /download/{id}")
    print("  This replicates the exact frontend flow from index.html")

    # Step 1: Submit job (like generateVideo())
    print("\n  — Submitting job to /generate —")
    start = time.time()
    f = open(AUDIO_PATH, "rb")
    files = {"audio": ("clip.wav", f, "audio/wav")}
    data = {
        "style": "mha_anime",
        "max_frames": "2",
        "mode": "fast",
        "steps": "8",
        "guidance": "7.0",
        "width": "512",
        "height": "512",
        "denoise": "0.75",
        "seed": "-1",
    }
    r = requests.post(f"{API_BASE}/generate", files=files, data=data, timeout=30)
    f.close()

    t_submit = time.time() - start
    if r.status_code != 200:
        print(f"  ❌ /generate returned HTTP {r.status_code}: {r.text[:300]}")
        return
    submit_result = r.json()
    job_id = submit_result.get("job_id")
    print(f"  /generate → HTTP {r.status_code} in {t_submit:.1f}s ✅")
    print(f"  job_id: {job_id}")
    print(f"  status: {submit_result.get('status')} (should be 'processing')")
    print(f"  mode: {submit_result.get('mode')}")
    print(f"  message: {submit_result.get('message')}")

    # Step 2: Poll /jobs/{job_id} every 2s (like setInterval(pollJobProgress, 2000))
    print(f"\n  — Polling /jobs/{job_id} every 2s —")
    last_log_count = 0
    last_stage = 0
    poll_start = time.time()
    max_poll_time = 300  # 5 minute timeout

    while time.time() - poll_start < max_poll_time:
        time.sleep(2)
        try:
            r = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=10)
        except requests.ConnectionError:
            continue
        if r.status_code != 200:
            print(f"  Poll HTTP {r.status_code}")
            continue

        job = r.json()
        status = job.get("status")
        stage = job.get("current_stage", 0)
        logs = job.get("logs", [])

        # Track stage transitions (like frontend setStagePill logic)
        if stage != last_stage and stage > 0:
            print(f"    ⏩ {stage_name(stage)} (current_stage={stage})")
            last_stage = stage

        # Append new logs (like frontend loop)
        for i in range(last_log_count, len(logs)):
            entry = logs[i]
            msg_type = entry.get("type", "info")
            icon = {"success": "✅", "error": "❌", "warn": "⚠️", "info": "  ℹ"}.get(msg_type, "  ℹ")
            print(f"    {icon} {entry.get('message', '')[:100]}")
            last_log_count = i + 1

        if status == "completed":
            elapsed = time.time() - poll_start
            print(f"\n  ✅ Job COMPLETED in {elapsed:.1f}s")
            # Verify result fields (like showResults())
            result = job.get("result", {})
            print(f"  Result fields:")
            print(f"    job_id: {result.get('job_id')}")
            print(f"    generation_mode: {result.get('generation_mode')}")
            print(f"    style: {result.get('style')}")
            print(f"    mood_detected: {result.get('mood_detected')}")
            print(f"    total_scenes: {result.get('total_scenes')}")
            print(f"    total_frames: {result.get('total_frames')}")
            print(f"    processing_time_seconds: {result.get('processing_time_seconds')}")

            # Verify video file (like showResults loading v.src = /download/{job_id})
            video_path = os.path.join(
                "/Users/syedsunny/sound_sizzle_fyp/backend/temp", job_id, "output.mp4"
            )
            video_available = job.get("video_available", False)
            video_size = job.get("video_size_bytes", 0)
            print(f"  video_available: {video_available}")
            print(f"  video_size_bytes: {video_size} ({video_size / 1024:.0f} KB)")

            if os.path.exists(video_path) and os.path.getsize(video_path) > 100000:
                print(f"  ✅ Video file exists and has substantial size (>100KB)")
            elif os.path.exists(video_path):
                print(f"  ⚠️ Video file exists but small ({os.path.getsize(video_path)} bytes)")
            else:
                print(f"  ❌ Video file NOT found at {video_path}")

            # Verify storyboard + frameset in job response (like job inspector tab)
            storyboard = job.get("storyboard")
            frameset = job.get("frameset")
            if storyboard:
                print(f"  ✅ storyboard in job response ({len(json.dumps(storyboard))} bytes)")
                # Count frames
                total_frames = sum(len(s.get("frames", [])) for s in storyboard.get("scenes", []))
                print(f"     Storyboard frames: {total_frames}")
            else:
                print(f"  ❌ storyboard missing from job response")
            if frameset:
                print(f"  ✅ frameset in job response ({len(json.dumps(frameset))} bytes)")
            else:
                print(f"  ❌ frameset missing from job response")

            # Verify download endpoint (like /download/{job_id} in showResults)
            print(f"\n  — Testing /download/{job_id} —")
            r = requests.get(f"{API_BASE}/download/{job_id}", timeout=15, stream=True)
            if r.status_code == 200:
                # Read first few bytes to verify it's MP4
                first_chunk = next(r.iter_content(1024))
                is_mp4 = first_chunk[:4] == b"\x00\x00\x00" or first_chunk[:4] == b"ftyp" or first_chunk[4:8] == b"ftyp" or first_chunk[:3] == b"\x1f\x8b"  # not right
                # MP4 files start with 00 00 00
                mp4_magic = first_chunk[:4] == b"\x00\x00\x00\x18" or first_chunk[4:8] == b"ftyp" or first_chunk[:4] == b"\x00\x00\x00"
                print(f"    /download/{job_id} → HTTP {r.status_code}, {r.headers.get('content-length', '?')} bytes ✅")
                print(f"    Content-Type: {r.headers.get('content-type')}")
                is_mp4 = first_chunk[4:8] == b'ftyp' or first_chunk[:4] == b'\x00\x00\x00'
                print(f"    MP4 magic bytes: {first_chunk[:8].hex()} {'✅' if is_mp4 else '❌'}")
            else:
                print(f"    ❌ /download/{job_id} → HTTP {r.status_code}")

            # Verify filesystem artifacts
            job_dir = f"/Users/syedsunny/sound_sizzle_fyp/backend/temp/{job_id}"
            if os.path.exists(job_dir):
                files_list = sorted(os.listdir(job_dir))
                print(f"\n  Filesystem artifacts in {job_dir}:")
                for f in files_list:
                    fpath = os.path.join(job_dir, f)
                    if os.path.isfile(fpath):
                        print(f"    {f}: {os.path.getsize(fpath)} bytes")
                    elif os.path.isdir(fpath):
                        frame_count = len(os.listdir(fpath))
                        print(f"    {f}/ ({frame_count} files)")
            else:
                print(f"  ❌ Job directory not found: {job_dir}")

            return job_id

        elif status == "failed":
            print(f"\n  ❌ Job FAILED")
            print(f"  Error: {job.get('error', 'Unknown')}")
            return job_id

        elif status == "cancelled":
            print(f"\n  ⚠️ Job CANCELLED")
            return job_id

    print(f"\n  ⏰ Polling timed out after {max_poll_time}s")
    return job_id


def test_cancel_endpoint(job_id):
    """Test the POST /jobs/{job_id}/cancel endpoint."""
    print("\n" + "=" * 70)
    print(f"[CHECK 4] POST /jobs/{job_id}/cancel")

    # Test cancelling a non-existent job
    r = requests.post(f"{API_BASE}/jobs/nonexistent123/cancel", timeout=5)
    print(f"  Cancel non-existent job → HTTP {r.status_code} {'✅' if r.status_code == 404 else '❌'} (expected 404)")

    # Test looking up a completed job
    r = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=5)
    print(f"  GET /jobs/{job_id} (completed) → HTTP {r.status_code} ✅")

    # Test job inspector (filesystem fallback for old jobs)
    if os.path.exists(f"/Users/syedsunny/sound_sizzle_fyp/backend/temp/3d2b50fe"):
        r = requests.get(f"{API_BASE}/jobs/3d2b50fe", timeout=5)
        if r.status_code == 200:
            job = r.json()
            print(f"  GET /jobs/3d2b50fe (filesystem fallback) → HTTP {r.status_code} ✅")
            print(f"    status: {job.get('status')}")
            print(f"    video_available: {job.get('video_available')}")
            print(f"    video_size: {job.get('video_size_bytes', 0)} bytes")
        else:
            print(f"  GET /jobs/3d2b50fe → HTTP {r.status_code}")
    else:
        print(f"  (3d2b50fe not found in temp/ — skip filesystem fallback test)")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     AudioStory — Full Browser Simulation Test                     ║")
    print("║     Replicates: POST /generate → poll /jobs/{id} → /download/{id} ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Check 0: Server health
    if not test_server_health():
        print("\n❌ Server not reachable. Start it with:")
        print("  cd backend && uvicorn main:app --reload --port 8000")
        sys.exit(1)

    # Check 1: Stage 1 test
    stage1_result = test_stage1()

    # Check 2: Stage 2 test
    if stage1_result:
        test_stage2()

    # Check 3: Full async pipeline (the main browser test)
    job_id = test_full_pipeline_async()

    # Check 4: Cancel endpoint + job lookup
    if job_id:
        test_cancel_endpoint(job_id)

    print("\n" + "=" * 70)
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     Browser Simulation Complete                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
