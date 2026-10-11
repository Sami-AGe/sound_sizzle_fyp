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

from models.schemas import AudioAnalysisResult
from utils.audio_utils import get_energy_at_time


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


def analyze_audio(
    audio_path: str,
    whisper_model_size: str = WHISPER_MODEL_SIZE,
    save_path: str = None
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

    # ── Mood Detection ───────────────────────────────────────────────────────
    avg_energy = float(np.mean(rms_normalized))
    detected_mood = detect_mood(avg_energy, sc_mean, tempo, zcr)
    print(f"[Stage 1] Mood detected: {detected_mood} (energy={avg_energy:.2f})")

    # ── Energy Sections (structural segments) ────────────────────────────────
    energy_sections = segment_energy(rms_normalized, rms_timestamps, duration)

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

    analysis = AudioAnalysisResult(
        duration=duration,
        tempo=tempo,
        beat_times=beat_times,
        onset_times=onset_times,
        rms_values=rms_normalized,
        rms_timestamps=rms_timestamps,
        spectral_centroid_mean=sc_mean,
        spectral_centroid_over_time=sc_over_time,
        chroma_mean=chroma_mean,
        mfcc_mean=mfcc_mean,
        zero_crossing_rate=zcr,
        transcript=transcript,
        detected_mood=detected_mood,
        energy_sections=energy_sections
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
