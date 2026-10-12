"""
Beat-aligned frame timing utilities.

Snaps frame transitions to real beat hits so that every visual cut
lands within ~33ms of a detected beat in the audio.
"""

from typing import List, Optional


def compute_beat_aligned_frame_durations(
    beat_times: List[float],
    frame_directives: List,
    total_duration: float,
    min_hold: float = 0.5,
    max_hold: float = 4.0,
) -> List[float]:
    """
    Compute hold durations so frame transitions land on real beat hits.

    For each frame *i* (except the last), the transition to the next frame
    is snapped to the beat hit closest to ``frame_directives[i+1].timestamp``.
    This ensures every frame boundary falls within one beat spacing of a
    real beat hit while preserving the storyboard's intended relative pacing.

    The last frame extends to ``total_duration`` (or the last beat before it),
    then absorbs any rounding drift so the durations sum to ~``total_duration``.

    Parameters
    ----------
    beat_times : list of float
        Beat timestamps from Stage 1 audio analysis.
    frame_directives : list
        FrameDirective objects (must have ``.timestamp`` attribute).
    total_duration : float
        Total audio duration in seconds.
    min_hold, max_hold : float
        Clamp range for each frame's hold duration.

    Returns
    -------
    list of float
        Hold durations (one per frame) that sum to ~``total_duration``.
    """
    durations: List[float] = []
    n = len(frame_directives)

    # Compute max snap distance: half the median beat spacing, default 0.75s
    if len(beat_times) > 1:
        spacings = [beat_times[i + 1] - beat_times[i] for i in range(len(beat_times) - 1)]
        max_snap_dist = sorted(spacings)[len(spacings) // 2] / 2
    else:
        max_snap_dist = 0.75

    for i, frame in enumerate(frame_directives):
        if i < n - 1:
            next_ts = frame_directives[i + 1].timestamp
            target_end = _nearest_beat(beat_times, next_ts)
            # Only snap if the nearest beat is reasonably close; otherwise
            # fall back to the original frame timestamp
            if target_end is None or abs(target_end - next_ts) > max_snap_dist:
                target_end = next_ts
            if target_end <= frame.timestamp:
                target_end = next_ts
            duration = target_end - frame.timestamp
        else:
            # Last frame: extend to last beat before total_duration, or total_duration
            target_end = _nearest_beat_before(beat_times, total_duration)
            if target_end is not None and target_end >= frame.timestamp:
                duration = target_end - frame.timestamp
            else:
                duration = total_duration - frame.timestamp

        duration = max(min_hold, min(max_hold, max(0.01, duration)))
        durations.append(round(duration, 3))

    # Absorb rounding / clamp drift into the last frame so durations ≈ total_duration
    if durations:
        drift = total_duration - sum(durations)
        if abs(drift) > 0.01:
            durations[-1] = round(max(0.01, durations[-1] + drift), 3)

    return durations


def _nearest_beat(beat_times: List[float], target: float) -> Optional[float]:
    """Return the beat hit closest to *target*."""
    if not beat_times:
        return None
    return min(beat_times, key=lambda b: abs(b - target))


def _nearest_beat_before(beat_times: List[float], target: float) -> Optional[float]:
    """Return the beat hit closest to *target* but not after it."""
    if not beat_times:
        return None
    before = [b for b in beat_times if b <= target]
    if before:
        return max(before, key=lambda b: b)
    return None
