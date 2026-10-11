"""
AudioStory — Audio Utility Functions

Helper functions for extracting audio features at specific timestamps,
used by both the pipeline and the storyboard builder.
"""

import numpy as np
import librosa


def get_energy_at_time(
    rms_values: list, rms_timestamps: list, t: float
) -> float:
    """
    Interpolates the RMS energy value at any given timestamp.
    Used to look up energy when a specific frame timestamp is given.
    """
    if not rms_timestamps or not rms_values:
        return 0.0
    idx = int(np.searchsorted(rms_timestamps, t))
    idx = min(idx, len(rms_values) - 1)
    idx = max(0, idx)
    return float(rms_values[idx])


def get_features_at_time(
    audio_data, t: float
) -> dict:
    """
    Returns all audio features at a specific timestamp.
    Convenience function that extracts energy, brightness, and onset proximity.
    """
    energy = get_energy_at_time(
        audio_data.rms_values, audio_data.rms_timestamps, t
    )
    brightness = get_brightness_at_time(
        audio_data.spectral_centroid_over_time,
        audio_data.rms_timestamps,
        t,
    )
    return {
        "energy": energy,
        "brightness": brightness,
        "is_onset": is_near_onset(audio_data.onset_times, t),
    }


def get_brightness_at_time(
    sc_over_time: list, rms_timestamps: list, t: float
) -> float:
    """
    Interpolates the spectral centroid (brightness) at a given timestamp.
    """
    if not rms_timestamps or not sc_over_time:
        return 0.0
    idx = int(np.searchsorted(rms_timestamps, t))
    idx = min(idx, len(sc_over_time) - 1)
    idx = max(0, idx)
    return float(sc_over_time[idx])


def is_near_onset(onset_times: list, t: float, window: float = 0.08) -> bool:
    """
    Returns True if timestamp t is within `window` seconds of any onset event.
    Default window is 80ms — roughly one video frame at 30fps.
    """
    return any(abs(t - o) <= window for o in onset_times)


def resample_rms_to_beat_times(
    rms_values: list, rms_timestamps: list, beat_times: list
) -> list:
    """
    Resamples the RMS energy series so there's one energy value per beat.
    This is useful for beat-driven visual effects.
    """
    result = []
    for t in beat_times:
        e = get_energy_at_time(rms_values, rms_timestamps, t)
        result.append(e)
    return result


def smooth_onsets(onset_times: list, rms_values: list, rms_timestamps: list) -> list:
    """
    Filters onset times to only include onsets at high-energy moments.
    Prevents visual effects from triggering on quiet background onsets.
    """
    significant = []
    for t in onset_times:
        e = get_energy_at_time(rms_values, rms_timestamps, t)
        if e > 0.2:
            significant.append(t)
    return sorted(significant)
