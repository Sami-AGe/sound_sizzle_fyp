"""
AudioStory — Data Schemas

These are the data contracts that flow between every stage. Every stage reads
from the previous stage's output schema and writes to the next one.
"""

from pydantic import BaseModel
from typing import List, Optional, Literal
from enum import Enum


# ─────────────────────────────────────────
# STAGE 1 OUTPUT — Audio Analysis Result
# ─────────────────────────────────────────

class BeatEvent(BaseModel):
    timestamp: float          # seconds from start
    strength: float           # 0.0 to 1.0

class OnsetEvent(BaseModel):
    timestamp: float          # seconds from start
    energy: float             # local energy at this moment

class AudioAnalysisResult(BaseModel):
    duration: float                    # total seconds
    tempo: float                       # BPM
    beat_times: List[float]            # list of beat timestamps in seconds
    onset_times: List[float]           # list of onset (hit/drop) timestamps
    rms_values: List[float]            # energy at each frame (normalized 0-1)
    rms_timestamps: List[float]        # timestamps corresponding to rms_values
    spectral_centroid_mean: float      # average tonal brightness (0-1 normalized)
    spectral_centroid_over_time: List[float]
    chroma_mean: List[float]           # 12 values, harmonic content
    mfcc_mean: List[float]             # 13 values, timbral texture
    zero_crossing_rate: float          # noisiness / vocal presence indicator
    transcript: Optional[str]          # lyrics if detected
    detected_mood: str                 # derived mood label (see mood detection)
    energy_sections: List[dict]        # [{start, end, avg_energy, label}]


# ─────────────────────────────────────────
# STAGE 2 OUTPUT — Storyboard (Groq)
# ─────────────────────────────────────────

class GenerationMode(str, Enum):
    FAST = "fast"
    QUALITY = "quality"

class CharacterAnchor(BaseModel):
    """Locked character + style descriptor generated in Stage 2 Call 1.
    Used verbatim in Fast mode prompts to maintain visual consistency."""
    hair: str
    eyes: str
    clothing: str
    body: str
    distinguishing: str
    full_anchor_text: str       # used verbatim in Fast mode prompts
    style_anchor: str           # style tokens, also used verbatim
    reference_prompt: str       # for Quality mode reference image generation

class EmotionalArc(BaseModel):
    """Narrative arc mapped to audio energy sections, used for Fast mode prompt guidance."""
    intro: str
    buildup: str
    climax: str
    outro: str

class FrameType(str, Enum):
    KEYFRAME = "KEYFRAME"
    CONTINUATION = "CONTINUATION"

class FrameDirective(BaseModel):
    frame_id: int
    timestamp: float
    type: FrameType
    # Only present on KEYFRAME:
    full_prompt: Optional[str] = None          # complete image generation prompt
    visual_state: Optional[str] = None         # KEYFRAME only (Fast mode): what persists into scene's continuations
    subject_pose: Optional[str] = None        # how the character is positioned
    camera_angle: Optional[str] = None        # wide shot / close up / medium / etc
    lighting: Optional[str] = None            # dramatic / soft / neon / etc
    # Only present on CONTINUATION:
    modifier_prompt: Optional[str] = None     # what changes from previous frame
    strength: Optional[float] = None          # img2img deviation strength 0.1-0.5
    what_changes: Optional[str] = None        # human readable change description

class Scene(BaseModel):
    scene_id: int
    scene_seed: Optional[int] = None   # Fast mode: seed for all frames in this scene
    start_time: float
    end_time: float
    narrative: str                     # what is happening in this scene
    setting: str                       # physical location / environment
    mood: str                          # emotional tone of this scene
    color_palette: str                 # visual color description
    transition_from_previous: str      # how this scene enters (cut/fade/glitch)
    frames: List[FrameDirective]

class Storyboard(BaseModel):
    generation_mode: GenerationMode
    main_story: str                    # overall narrative in 2-3 sentences
    total_scenes: int
    scenes: List[Scene]

    # Fast mode fields
    character_anchor: Optional[CharacterAnchor] = None
    style_anchor: Optional[str] = None
    emotional_arc: Optional[EmotionalArc] = None

    # Quality mode fields (Optional so Fast mode output is valid)
    subject_description: Optional[str] = None
    subject_reference_prompt: Optional[str] = None


# ─────────────────────────────────────────
# STAGE 3 OUTPUT — Generated Frames
# ─────────────────────────────────────────

class GeneratedFrame(BaseModel):
    frame_id: int
    timestamp: float
    type: FrameType
    image_path: str                    # local path to generated .png
    duration: float                    # how long this frame holds in the video
    scene_id: int
    prompt_used: str
    strength_used: Optional[float] = None

class GeneratedFrameSet(BaseModel):
    job_id: str
    style_name: str
    reference_image_path: Optional[str] = None   # Quality mode only
    frames: List[GeneratedFrame]
    total_frames: int
    generation_mode: str = "fast"                # "fast" or "quality"


# ─────────────────────────────────────────
# STAGE 4 OUTPUT — Final Video
# ─────────────────────────────────────────

class FinalVideo(BaseModel):
    job_id: str
    output_path: str
    duration: float
    total_frames_used: int
    effects_applied: List[str]
    audio_source: str
