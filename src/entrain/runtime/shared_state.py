"""The shared control slot. [BUILD — the decoupling primitive]

A single atomically-updated slot holding the current control: the StyleVector plus a
separate TEMPO scalar (tempo travels alongside the style vector, never inside it — the
embedding controls character, not BPM). The slow VISION loop writes it; the fast AUDIO
loop reads it. No locks on the audio read path — worst case is reading a slightly stale
value, never blocking. This one object keeps the camera from ever stalling the sound.

It also rate-limits the style vector and smooths tempo so the music morphs instead of
twitching — what reaches the engine is already tamed.
"""

from __future__ import annotations

import threading

import numpy as np

from entrain.types import StyleVector


class StyleSlot:
    """Thread-safe latest-value slot with write-side smoothing / rate limiting."""

    def __init__(self, max_step: float = 0.12, tempo_alpha: float = 0.1,
                 responsive: bool = True) -> None:
        # max_step: max L2 movement of the style vector per write — a pure anti-TWITCH
        # guard against single-frame detection noise. The "hold the vibe" dynamics
        # (fast rise / slow fall) live in StateEstimator, so keep this loose enough that
        # it doesn't symmetrically throttle the rise.
        self.max_step = max_step
        self.tempo_alpha = tempo_alpha     # EMA smoothing on tempo (avoid BPM jitter)
        # responsive=False -> the STATIC-generative baseline for A/B: the first style
        # (and tempo) writes through to establish a starting point, then the slot is
        # frozen and ignores all movement-driven updates. Tests "is the closed loop more
        # engaging than fixed generative music?" — the loop runs identically, only the
        # steering is severed. Onset/melody accents are suppressed too.
        self.responsive = responsive
        self._lock = threading.Lock()
        self._current: StyleVector | None = None
        self._tempo = 0.0
        self._onset = False
        self._drum_hit = False      # latched for the HUD so it never misses a hit
        self._melody_pitch: int | None = None   # current lead pitch (None = not playing)

    def write(self, style: StyleVector, tempo: float | None = None) -> None:
        """Called by the vision loop. Style rate-limited; tempo EMA-smoothed."""
        with self._lock:
            if self._current is None:
                self._current = style
            elif not self.responsive:
                return                        # static baseline: hold the first style
            else:
                delta = style.vec - self._current.vec
                dist = float(np.linalg.norm(delta))
                if dist > self.max_step:
                    delta = delta * (self.max_step / dist)
                self._current = StyleVector(
                    vec=self._current.vec + delta, weights=style.weights
                )
            if tempo is not None and tempo > 1.0:
                a = self.tempo_alpha
                self._tempo = a * tempo + (1 - a) * self._tempo if self._tempo > 1 else tempo

    def read(self) -> StyleVector | None:
        """Called by the audio loop every chunk. Lock is held only for a pointer copy."""
        with self._lock:
            return self._current

    @property
    def tempo(self) -> float:
        with self._lock:
            return self._tempo

    def set_onset(self) -> None:
        """Vision loop: a movement accent happened (sticky until consumed)."""
        with self._lock:
            if self.responsive:
                self._onset = True

    def take_onset(self) -> bool:
        """Audio loop: read and clear the pending accent."""
        with self._lock:
            v = self._onset
            self._onset = False
            return v

    def set_melody(self, pitch: int | None) -> None:
        """Vision loop: the lead pitch the hand is currently playing (None = silent)."""
        with self._lock:
            if self.responsive:
                self._melody_pitch = pitch

    @property
    def melody(self) -> int | None:
        with self._lock:
            return self._melody_pitch

    def set_drum(self, value: int) -> None:
        """Audio loop: the drum value the engine just played (1 = a hit). Latches hits."""
        with self._lock:
            if value == 1:
                self._drum_hit = True

    def take_drum_hit(self) -> bool:
        """HUD: did a drum hit happen since the last check? (read-and-clear)"""
        with self._lock:
            v = self._drum_hit
            self._drum_hit = False
            return v
