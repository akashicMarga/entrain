"""Shared data contracts — the seams between pipeline stages.

Every module communicates only through these types. As long as a module honors the
type it produces/consumes, it can be swapped freely. This is what makes the two big
forks cheap: selection -> generation (`generation/`) and heuristic -> learned policy
(`policy/`). See docs/architecture.md for the full table.

Nothing here is trainable. These are plain dataclasses carrying numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --- capture --------------------------------------------------------------------

@dataclass(slots=True)
class Frame:
    """One RGB frame from the camera. [world -> perception]"""

    rgb: np.ndarray          # (H, W, 3) uint8
    t: float                 # capture timestamp, seconds (monotonic)
    frame_id: int


# --- perception features --------------------------------------------------------

@dataclass(slots=True)
class PoseFeatures:
    """Aggregated movement, never a raw skeleton. [perception.pose -> state]"""

    motion_energy: float     # 0..1, overall (gross) movement magnitude
    movement_tempo: float    # estimated BPM of motion, or nan
    synchrony: float         # 0..1, cross-subject movement correlation (~10s window)
    lead_hand_height: float = 0.0   # 0=down .. 1=raised, the higher wrist (for melody)
    micromotion: float = 0.0        # 0..1, subtle-movement intensity of a near-still
                                    #   listener — the signal that survives when gross
                                    #   motion_energy ~ 0 (seated, quietly engaged)


@dataclass(slots=True)
class AffectFeatures:
    """Crowd-averaged affect. [perception.affect -> state]"""

    valence: float           # -1..1 (unpleasant..pleasant)
    arousal: float           # 0..1 (calm..activated)
    n_faces: int


@dataclass(slots=True)
class CardiacFeatures:
    """Contactless heart rate, close-range single-face only. [perception.rppg -> state]"""

    heart_rate: float        # BPM, or nan if signal too weak
    hr_trend: float          # signed slope over recent window
    confidence: float        # 0..1, drops under motion/low-light/distance


# --- fused state ----------------------------------------------------------------

@dataclass(slots=True)
class StateVector:
    """Tiny human-readable summary of the crowd. Internal — never sent to the generator.

    [state.estimator -> policy]
    """

    arousal: float           # 0..1
    valence: float           # -1..1
    energy: float            # 0..1
    synchrony: float         # 0..1  <- the impact metric (not average arousal)
    tempo: float             # BPM
    heart_rate: float = 0.0  # BPM, display/context only — NOT part of as_array
    hr_tension: float = 0.0  # -1..1, heart rate vs personal baseline (build/tension)

    def as_array(self) -> np.ndarray:
        # The 5 steering dims only. heart_rate is contextual (noisy, person-relative),
        # carried alongside for display rather than fed to the policy/smoother.
        return np.array(
            [self.arousal, self.valence, self.energy, self.synchrony, self.tempo],
            dtype=np.float32,
        )


# --- director (slow planning layer) ---------------------------------------------

@dataclass(slots=True)
class AnchorSpec:
    """One anchor prompt the Director authors. [policy.director -> policy.anchors]"""

    name: str
    prompt: str


@dataclass(slots=True)
class Directive:
    """The slow DIRECTOR layer's output: which anchors define the current musical range,
    plus a human-readable intent (the set's arc/goal). Re-authored on a seconds-scale
    cadence, NEVER per frame — the Director is the planner, not the controller. Anchors are
    ordered low-energy -> high-energy (the fast policy assumes that axis).

    [policy.director -> policy.anchors / policy]
    """

    anchors: list[AnchorSpec]      # spans the intended range, low -> high energy
    intent: str = ""               # e.g. "warm, beatless build" — logging + LLM memory


@dataclass(slots=True)
class DirectorContext:
    """The slow, aggregated view of the room the Director reasons over (seconds-scale),
    plus an optional natural-language instruction. Deliberately NOT per-frame — this is the
    planner's input, distilled from many `StateVector`s. [runtime -> policy.director]
    """

    elapsed_s: float               # seconds since the set started
    energy_mean: float             # recent-window aggregates (~10-30 s)
    energy_trend: float            # signed slope of energy over the window
    valence_mean: float
    synchrony_mean: float
    seconds_in_directive: float    # how long the current Directive has been live
    instruction: str = ""          # natural-language steer from the operator (optional)
    last_response_synchrony: float = 0.0   # Δsynchrony AFTER the director's last move — did
                                           #   the room get more together? (the feedback signal)


# --- policy / generation interface ----------------------------------------------

@dataclass(slots=True)
class AnchorSet:
    """Anchor prompts embedded once into the generator's shared style space.

    [policy.anchors -> policy]
    """

    names: list[str]         # e.g. ["calm ambient", "groovy house", "peak techno"]
    embeddings: np.ndarray   # (k, d) one row per anchor


@dataclass(slots=True)
class StyleVector:
    """THE signal on the wire. A blend of anchor embeddings. [policy -> generation]

    Carries the blended vector plus the human-readable weights that produced it (for
    logging/debugging). Carries NO tempo (style controls character, not BPM) and is
    expected to arrive already smoothed/rate-limited.
    """

    vec: np.ndarray                                   # (d,) the blended style embedding
    weights: dict[str, float] = field(default_factory=dict)


# --- audio ----------------------------------------------------------------------

@dataclass(slots=True)
class AudioChunk:
    """One block of PCM from the generator. [generation -> output]"""

    pcm: np.ndarray          # (n_samples, n_channels) float32 in [-1, 1]
    sample_rate: int
