"""
AudioStory — Stage 1: Audio Analysis

Analyzes a 30-second audio clip and extracts a complete emotional and
structural fingerprint. Runs entirely on CPU with no external API calls.

Outputs: AudioAnalysisResult (with JSON serialization to disk)
"""

import os
import json
import numpy as np
import librosa
import whisper

from groq import Groq
from models.schemas import AudioAnalysisResult
from utils.audio_utils import get_brightness_at_time

# Valid mood keys for the MOOD_GRADES registry (stage2/stage4)
VALID_MOODS = {"melancholic", "energetic", "dark_intense", "euphoric", "tense", "neutral", "neutral_cinematic"}


# Default model sizes for whisper transcription.
# tiny = fastest, base = balanced, small/medium = higher accuracy
WHISPER_MODEL_SIZE = "base"


def detect_mood(energy: float, brightness: float, tempo: float, zcr: float) -> str:
    """
    Derives a single mood label from combined audio features.
    This label is passed to Groq to set the narrative tone.

    Thresholds determine emotional classification:
    - High energy + fast tempo → intense/aggressive
    - High energy + moderate tempo → energetic/euphoric
    - Moderate energy + bright → uplifting/heroic
    - Moderate energy + dark → dark/powerful
    - Low energy + dark → melancholic/lonely
    - Low energy + bright → peaceful/dreamy
    - High zero-crossing rate → chaotic/anxious
    """
    if energy > 0.7 and tempo > 130:
        return "intense_aggressive"
    elif energy > 0.7 and tempo > 100:
        return "energetic_euphoric"
    elif energy > 0.5 and brightness > 0.6:
        return "uplifting_heroic"
    elif energy > 0.5 and brightness < 0.4:
        return "dark_powerful"
    elif energy < 0.3 and brightness < 0.4:
        return "melancholic_lonely"
    elif energy < 0.3 and brightness > 0.6:
        return "peaceful_dreamy"
    elif zcr > 0.1:
        return "chaotic_anxious"
    else:
        return "neutral_cinematic"


def segment_energy(rms: list, timestamps: list, duration: float) -> list:
    """
    Splits the audio into 4 structural sections (intro/buildup/climax/outro)
    and labels each by energy level. This tells Groq when to place
    scene boundaries.
    """
    n = len(rms)
    section_size = max(1, n // 4)
    sections = []

    labels = ["intro", "buildup", "climax", "outro"]
    for i, label in enumerate(labels):
        start_idx = i * section_size
        end_idx = start_idx + section_size if i < 3 else n
        if start_idx >= n:
            break
        section_rms = rms[start_idx:end_idx]
        avg = float(np.mean(section_rms)) if section_rms else 0.0
        end_time = timestamps[min(end_idx - 1, len(timestamps) - 1)]
        sections.append({
            "label": label,
            "start": float(timestamps[start_idx]),
            "end": float(end_time),
            "avg_energy": avg,
            "energy_label": "high" if avg > 0.6 else "medium" if avg > 0.3 else "low"
        })
    return sections


def _detect_mood(transcript: str, tempo: float, avg_energy: float, groq_client) -> str:
    """
    Uses Groq LLM to classify the song's mood into one of the MOOD_GRADES keys.
    Falls back to audio-feature-based detect_mood() when no Groq client is available.
    """
    if groq_client is None:
        # Fallback: map audio-feature moods to MOOD_GRADES keys
        audio_mood = detect_mood(avg_energy, 0.5, tempo, 0.05)
        mood_map = {
            "intense_aggressive": "dark_intense",
            "energetic_euphoric": "energetic",
            "uplifting_heroic": "euphoric",
            "dark_powerful": "dark_intense",
            "melancholic_lonely": "melancholic",
            "peaceful_dreamy": "melancholic",
            "chaotic_anxious": "tense",
            "neutral_cinematic": "neutral_cinematic",
        }
        return mood_map.get(audio_mood, "neutral_cinematic")

    prompt = (
        f"Song lyrics: {transcript[:500]}\n"
        f"Tempo: {tempo:.0f} BPM, Energy: {'high' if avg_energy > 0.05 else 'low'}\n\n"
        "Classify mood as one of: melancholic / energetic / dark_intense / euphoric / tense / neutral / neutral_cinematic\n"
        "Respond with one word only."
    )
    try:
        resp = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        mood = resp.choices[0].message.content.strip().lower().replace("-", "_")
        return mood if mood in VALID_MOODS else "neutral"
    except Exception:
        return "neutral"


def analyze_audio(
    audio_path: str,
    whisper_model_size: str = WHISPER_MODEL_SIZE,
    save_path: str = None,
    max_frames: int = None
) -> AudioAnalysisResult:
    """
    Main entry point: load audio, extract all features, transcribe, and
    return an AudioAnalysisResult.

    Parameters
    ----------
    audio_path : str
        Path to the audio file (mp3, wav, m4a, ogg).
    whisper_model_size : str
        Whisper model size: 'tiny', 'base', 'small', or 'medium'.
    save_path : str, optional
        If provided, saves the result as JSON to this path.
    max_frames : int, optional
        Target total frame count. Used to scale unique-moments detection
        so more distinctive moments are identified for higher frame budgets.

    Returns
    -------
    AudioAnalysisResult
        Complete audio fingerprint including beats, onsets, energy, mood, etc.
    """
    print(f"[Stage 1] Loading audio: {audio_path}")

    # ── Load audio ──────────────────────────────────────────────────────────
    y, sr = librosa.load(audio_path, duration=30, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    print(f"[Stage 1] Duration: {duration:.2f}s, Sample rate: {sr}Hz")

    # ── Tempo and beats ──────────────────────────────────────────────────────
    # librosa 0.11 returns tempo as float when aggregate is default
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
    tempo = float(tempo) if hasattr(tempo, '__iter__') else float(tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    print(f"[Stage 1] Tempo: {tempo:.1f} BPM, Beats detected: {len(beat_times)}")

    # ── Onset detection (hits and drops) ────────────────────────────────────
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr,
        units='time',
        delta=0.2,          # sensitivity — lower = more onsets detected
        wait=5              # minimum gap between onsets (frames)
    )
    onset_times = onset_frames.tolist()
    print(f"[Stage 1] Onsets detected: {len(onset_times)}")

    # ── RMS Energy over time ────────────────────────────────────────────────
    hop_length = 512
    frame_length = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Normalize to 0-1 range safely
    rms_max = rms.max() if rms.max() > 0 else 1.0
    rms_normalized = (rms / rms_max).tolist()
    rms_timestamps = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop_length
    ).tolist()

    # ── Spectral features ───────────────────────────────────────────────────
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sc_max = spectral_centroid.max() if spectral_centroid.max() > 0 else 1.0
    sc_normalized = (spectral_centroid / sc_max)
    sc_mean = float(np.mean(sc_normalized))
    sc_over_time = sc_normalized.tolist()

    # ── Chroma (harmonic content, 12 pitch classes) ──────────────────────────
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = [float(v) for v in np.mean(chroma, axis=1)]

    # ── MFCCs (timbral texture) ──────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = [float(v) for v in np.mean(mfcc, axis=1)]

    # ── Zero crossing rate (noisiness) ───────────────────────────────────────
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

    # ── Transcription (Whisper) ──────────────────────────────────────────────
    transcript = None
    try:
        print(f"[Stage 1] Loading Whisper model: {whisper_model_size}")
        model = whisper.load_model(whisper_model_size)
        result = model.transcribe(audio_path)
        transcript = result["text"].strip()
        print(f"[Stage 1] Transcript: {transcript[:80]}{'...' if len(transcript) > 80 else ''}")
    except Exception as e:
        print(f"[Stage 1] Whisper transcription failed: {e}")

    # ── Mood Detection ───────────────────────────────────────────────────────
    avg_energy = float(np.mean(rms_normalized))
    # Use Groq-based mood detection if API key available, else fallback to audio features
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
    detected_mood = _detect_mood(transcript or "", tempo, avg_energy, groq_client)
    print(f"[Stage 1] Mood detected: {detected_mood} (energy={avg_energy:.2f})")

    # ── Energy Sections (structural segments) ────────────────────────────────
    energy_sections = segment_energy(rms_normalized, rms_timestamps, duration)

    # ── Detect unique moments for keyframe placement ─────────────────────────
    # Create a temporary AudioAnalysisResult to pass to detect_unique_moments
    _analysis_stub = AudioAnalysisResult(
        duration=duration,
        tempo=tempo,
        beat_times=beat_times,
        onset_times=onset_times,
        rms_values=rms_normalized,
        rms_timestamps=rms_timestamps,
        spectral_centroid_mean=sc_mean,
        spectral_centroid_over_time=sc_normalized.tolist(),
        chroma_mean=chroma_mean,
        mfcc_mean=mfcc_mean,
        zero_crossing_rate=zcr,
        transcript=transcript,
        detected_mood=detected_mood,
        energy_sections=energy_sections,
    )
    # Detect extra scenes worth of moments so the storyboard can pick keyframes
    # from the most distinctive audio points. Pass max_frames so the moment
    # count scales with the target frame resolution.
    # Use max(max_frames, 30) so short audio gets enough unique moments even if
    # max_frames isn't provided yet — Stage 1 runs before Stage 2 knows the
    # target, so we scale the moment count to the configured MAX_FRAMES env var.
    _max_frames_eff = max_frames or int(os.environ.get("MAX_FRAMES", "30"))
    _num_scenes_for_moments = determine_scene_count(_analysis_stub, max_frames=_max_frames_eff)
    detect_unique_moments(_analysis_stub, num_scenes=_num_scenes_for_moments)

    analysis = AudioAnalysisResult(
        duration=duration,
        tempo=tempo,
        beat_times=beat_times,
        onset_times=onset_times,
        rms_values=rms_normalized,
        rms_timestamps=rms_timestamps,
        spectral_centroid_mean=sc_mean,
        spectral_centroid_over_time=sc_normalized.tolist(),
        chroma_mean=chroma_mean,
        mfcc_mean=mfcc_mean,
        zero_crossing_rate=zcr,
        transcript=transcript,
        detected_mood=detected_mood,
        energy_sections=energy_sections,
        spectral_flux=_analysis_stub.spectral_flux,
        novelty_scores=_analysis_stub.novelty_scores,
        unique_moments=_analysis_stub.unique_moments,
    )

    # ── Save to disk ─────────────────────────────────────────────────────────
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(analysis.model_dump(), f, indent=2)
        print(f"[Stage 1] Analysis saved to: {save_path}")

    print("[Stage 1] Complete.")
    return analysis


def determine_scene_count(
    audio_data: AudioAnalysisResult,
    max_frames: int = None
) -> int:
    """
    Determines how many distinct scenes the storyboard should have
    based on energy sections and overall tempo.
    A 30-second video should have 3-5 scenes.

    If max_frames is provided, scene count is capped so that the LLM
    can distribute the target frame count across scenes.
    """
    high_energy_sections = [
        s for s in audio_data.energy_sections if s["energy_label"] == "high"
    ]
    if len(high_energy_sections) >= 3:
        scene_count = 5
    elif len(high_energy_sections) >= 2:
        scene_count = 4
    else:
        scene_count = 3

    # If a target frame count is specified, cap scenes to ensure enough
    # frames per scene for a plausible storyboard. Each scene needs at least
    # 1 keyframe + 1 continuation = 2 minimum frames.
    if max_frames is not None and max_frames > 0:
        # Reserve 1 frame per scene for the keyframe; the rest are continuations
        # Ensure at least 2 frames per scene (1 KF + 1 CONT) unless max_frames
        # is extremely low (then allow 1 frame/scene = all keyframes)
        min_frames_per_scene = 2 if max_frames >= scene_count * 2 else 1
        max_scenes = max_frames // min_frames_per_scene
        scene_count = min(scene_count, max_scenes)
        # Guarantee at least 1 scene
        scene_count = max(scene_count, 1)

    return scene_count


def detect_unique_moments(
    audio_data: AudioAnalysisResult,
    num_scenes: int = 3,
) -> list:
    """
    Identifies the most distinctive moments in the audio for keyframe placement.

    Computes a combined novelty score that weights:
    - **Spectral flux** (timbral/transient change) — sharp changes in brightness
    - **Onset proximity** — peaks near onsets (drops/hits) are unique
    - **Energy deviation** — frames with energy well above the local average

    Then uses scipy's find_peaks to locate local maxima, ranks them by
    combined score, and returns the top ``num_scenes + 2`` moments.

    Falls back to evenly-spaced beats if no peaks are found.

    Returns
    -------
    list of dict
        Each dict: ``{timestamp, energy, brightness, description}``
    """
    from scipy.signal import find_peaks as _scipy_find_peaks

    rms_vals = audio_data.rms_values
    rms_ts = audio_data.rms_timestamps
    onset_times = audio_data.onset_times or []
    sc_over_time = audio_data.spectral_centroid_over_time or []

    if not rms_vals or not rms_ts:
        return []

    # ── Compute spectral flux (derivative of spectral centroid) ──────────
    spectral_flux = []
    for i in range(len(sc_over_time)):
        if i == 0:
            spectral_flux.append(abs(sc_over_time[0] * 0.5))
        else:
            spectral_flux.append(abs(sc_over_time[i] - sc_over_time[i - 1]))
    audio_data.spectral_flux = [float(v) for v in spectral_flux]

    # ── Compute novelty score ──────────────────────────────────────────────
    # Combine normalized spectral flux (timbral change), energy deviation
    # (how far above the LOCAL mean), and onset proximity to identify moments
    # where the audio genuinely shifts — not just any peak.
    mean_flux = float(np.mean(spectral_flux)) if spectral_flux else 0.0

    novelty_scores = []
    for i, t in enumerate(rms_ts):
        # Energy deviation: how far above the running mean is this frame?
        # Use local mean (last 5-10 frames) to catch relative spikes, not just global peaks
        local_window = rms_vals[max(0, i - 5):i + 1]
        local_mean = float(np.mean(local_window)) if local_window else 0.0
        energy_dev = rms_vals[i] / (local_mean + 0.01) if local_mean > 0 else 0.0
        energy_dev = float(np.clip(energy_dev, 0, 3)) / 3.0  # normalize to 0-1

        # Spectral flux: timbral change
        flux_norm = spectral_flux[i] / (mean_flux + 0.01) if mean_flux > 0 else 0.0
        flux_norm = float(np.clip(flux_norm, 0, 3)) / 3.0

        # Onset proximity: is there an onset nearby? (strong indicator of a hit/drop)
        onset_near = any(abs(t - o) < 0.15 for o in onset_times)
        onset_score = 1.0 if onset_near else 0.0

        # Combine — flux and energy deviation weighted equally, onset as boost
        score = 0.35 * flux_norm + 0.35 * energy_dev + 0.30 * onset_score
        novelty_scores.append(float(score))

    audio_data.novelty_scores = novelty_scores

    # ── Detect peaks in novelty curve ──────────────────────────────────────
    if len(novelty_scores) < 3:
        peaks = []
    else:
        min_dist = max(3, int(0.3 * len(rms_ts) / audio_data.duration)) if audio_data.duration > 0 else 3
        # Raise prominence to avoid catching noise-level peaks
        min_prominence = 0.15
        peaks, _ = _scipy_find_peaks(
            novelty_scores,
            distance=min_dist,
            prominence=min_prominence,
        )

    # ── Select unique moments with precedence-based flow control ────────────
    # Goal: return enough candidate timestamps for keyframe placement that:
    #   1. They are the most "unique" (highest novelty score)
    #   2. They are well-spread across the audio timeline (no clustering)
    #   3. The count adapts to both num_scenes and audio complexity
    #
    # Precedence when there are too many peaks:
    #   - Cluster peaks within a time window (min_scene_gap ~ duration / num_scenes / 2)
    #   - From each cluster, keep only the highest-scoring peak
    #   - This prevents multiple keyframes from landing in the same audio segment
    #     which would disrupt visual flow with too-frequent cuts.
    #
    # Precedence when there are too few peaks:
    #   - Fall back to beat-aligned timestamps, prioritizing onsets
    #   - Spread them evenly across the duration so scenes still have keyframes

    target_count = max(num_scenes * 2, 5)

    if len(peaks) > 0:
        # Build (score, index, timestamp) tuples and sort by score descending
        peak_data = [(novelty_scores[p], p, float(rms_ts[p])) for p in peaks]

        # Cluster nearby peaks: group peaks within min_scene_gap seconds of each other
        min_scene_gap = audio_data.duration / (num_scenes * 2) if num_scenes > 0 else 5.0
        min_scene_gap = max(min_scene_gap, 3.0)  # At least 3s between cluster centers

        clustered_peaks = []
        for score, idx, ts in peak_data:
            # Find if this peak falls within min_scene_gap of any existing cluster center
            placed = False
            for cluster in clustered_peaks:
                if abs(ts - cluster["center_ts"]) < min_scene_gap:
                    # This peak is close to an existing cluster — keep the higher-scoring one
                    if score > cluster["best_score"]:
                        cluster["best_score"] = score
                        cluster["best_idx"] = idx
                        cluster["best_ts"] = ts
                    cluster["center_ts"] = (cluster["center_ts"] * cluster["count"] + ts) / (cluster["count"] + 1)
                    cluster["count"] += 1
                    placed = True
                    break
            if not placed:
                clustered_peaks.append({
                    "best_score": score, "best_idx": idx, "best_ts": ts,
                    "center_ts": ts, "count": 1
                })

        # Sort clusters by best score, take top target_count
        clustered_peaks.sort(key=lambda c: c["best_score"], reverse=True)
        selected_clusters = clustered_peaks[:target_count]
        # Sort back by time for chronological order
        selected_clusters.sort(key=lambda c: c["best_ts"])
        selected_peaks = [c["best_idx"] for c in selected_clusters]
    else:
        # Fallback: evenly-spaced beats, prioritizing onsets
        n_beats = len(audio_data.beat_times)
        if n_beats >= target_count:
            step = n_beats // target_count
            selected_peaks = list(range(0, n_beats, step))[:target_count]
        elif n_beats > 0:
            selected_peaks = list(range(n_beats))
        else:
            return []

    # ── Build unique moments list ──────────────────────────────────────────
    moments = []
    for idx in selected_peaks:
        if idx >= len(rms_ts):
            continue
        t = float(rms_ts[idx])
        energy = float(rms_vals[idx])
        brightness = get_brightness_at_time(
            sc_over_time, rms_ts, t
        ) if sc_over_time else audio_data.spectral_centroid_mean

        # Generate a description
        desc_parts = []
        if energy > 0.7:
            desc_parts.append("high-energy")
        elif energy > 0.4:
            desc_parts.append("moderate-energy")
        else:
            desc_parts.append("low-energy")
        if brightness > 0.6:
            desc_parts.append("bright timbre")
        elif brightness < 0.4:
            desc_parts.append("dark timbre")
        onset_near = any(abs(t - o) < 0.1 for o in onset_times)
        if onset_near:
            desc_parts.append("onbeat onset")

        moments.append({
            "timestamp": round(t, 2),
            "energy": round(energy, 2),
            "brightness": round(brightness, 2),
            "description": " + ".join(desc_parts),
        })

    audio_data.unique_moments = moments
    return moments


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m backend.stages.stage1_audio <audio_path> [whisper_model_size]")
        sys.exit(1)

    audio_file = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "base"

    result = analyze_audio(audio_file, whisper_model_size=model_size,
                           save_path=f"./temp/{os.path.basename(audio_file)}_analysis.json")

    # Print summary
    print(f"\n{'='*40}")
    print(f"Analysis Summary:")
    print(f"  Duration:          {result.duration:.1f}s")
    print(f"  Tempo:             {result.tempo:.1f} BPM")
    print(f"  Mood:              {result.detected_mood}")
    print(f"  Beats:             {len(result.beat_times)}")
    print(f"  Onsets:            {len(result.onset_times)}")
    print(f"  Energy sections:   {len(result.energy_sections)}")
    print(f"  Transcript:        {result.transcript[:100] if result.transcript else 'N/A'}")
    print(f"{'='*40}")
