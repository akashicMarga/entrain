"""Fusion -> state vector. [BUILD — mostly logic, no heavy training]

Fuses the three perception streams and SMOOTHS them into a compact `StateVector`.

The impact metric is SYNCHRONY (inter-subject correlation over ~10s windows), not average
arousal — a loud bad drop spikes arousal but not synchrony. Movement synchrony comes from
the camera; physiological synchrony is validated on the wearable subset.
"""

from __future__ import annotations

import numpy as np

from entrain.types import (
    AffectFeatures,
    CardiacFeatures,
    PoseFeatures,
    StateVector,
)


class StateEstimator:
    """Combine pose/affect/cardiac into a smoothed `StateVector`.

    Smoothing is ASYMMETRIC — like a DJ riding the room: quick to lift when movement
    rises (``attack``), slow to let it down when the room pauses (``release``). This is
    the "hold the vibe" inertia: a brief stillness no longer collapses straight to calm,
    it eases down over a few seconds. Per-field, so a momentary dip doesn't yank the music.
    """

    def __init__(self, attack: float = 0.45, release: float = 0.04,
                 hr_range: float = 12.0, micro_weight: float = 0.6) -> None:
        # attack: blend weight toward a RISING value (responsive).
        # release: blend weight toward a FALLING value (sluggish -> holds the vibe).
        self.attack = attack
        self.release = release
        self.hr_range = hr_range            # bpm above/below baseline that maps to ±1 tension
        self.micro_weight = micro_weight    # how much a still listener's subtle movement counts
        self._prev: StateVector | None = None
        self._hr_baseline = 0.0             # slow EMA of resting heart rate

    def update(
        self,
        pose: PoseFeatures,
        affect: AffectFeatures | None = None,
        cardiac: CardiacFeatures | None = None,
    ) -> StateVector:
        """Fuse the latest features. Affect/cardiac may be absent at crowd distance."""
        # Movement drive = gross motion OR (scaled) subtle micromotion, whichever is
        # stronger. A dancer is driven by motion_energy; a seated, quietly-engaged
        # listener (whom gross motion reads as ~0) is kept alive by micromotion.
        move = max(pose.motion_energy, self.micro_weight * pose.micromotion)
        arousal = move
        if affect is not None:
            arousal = 0.5 * arousal + 0.5 * affect.arousal
        valence = affect.valence if affect is not None else 0.0
        energy = move
        synchrony = pose.synchrony
        tempo = pose.movement_tempo

        raw = StateVector(arousal, valence, energy, synchrony, tempo)
        smoothed = self._smooth(raw)
        # heart_rate rides alongside (not smoothed via as_array); carry forward when the
        # rPPG signal drops out so the HUD doesn't flicker to 0.
        if cardiac is not None and cardiac.confidence > 0 and np.isfinite(cardiac.heart_rate):
            smoothed.heart_rate = cardiac.heart_rate
            # Slow baseline of resting HR; tension = how far above/below it we are now.
            if self._hr_baseline <= 0:
                self._hr_baseline = cardiac.heart_rate
            else:
                self._hr_baseline = 0.99 * self._hr_baseline + 0.01 * cardiac.heart_rate
            smoothed.hr_tension = float(
                np.clip((cardiac.heart_rate - self._hr_baseline) / self.hr_range, -1.0, 1.0)
            )
        elif self._prev is not None:
            smoothed.heart_rate = self._prev.heart_rate   # hold last; tension decays to 0
        self._prev = smoothed
        return smoothed

    def _smooth(self, raw: StateVector) -> StateVector:
        if self._prev is None:
            return raw
        prev = self._prev.as_array()
        cur = raw.as_array()
        # Per-field: fast alpha where the value is rising, slow alpha where it's falling.
        alpha = np.where(cur >= prev, self.attack, self.release)
        blend = alpha * cur + (1 - alpha) * prev
        return StateVector(*blend.tolist())
