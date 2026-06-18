"""Movement onset detector. [BUILD — no training]

Fires when motion energy spikes — a "hit"/accent gesture (punching the air, a sharp bob).
Runs on the RAW per-frame motion energy (before the state estimator's slow smoothing,
which would blunt accents). A refractory window prevents a single gesture from
machine-gunning multiple triggers.

A pure rising-edge detector over whatever signal it's fed: the runtime feeds it a
CALIBRATED level (movement above your resting floor, in spread units — see
state.calibration.SignalCalibrator), so ``threshold``/``floor`` are in those units, not
absolute energy.

Used to drive drum ACCENTS on top of the tempo-locked pulse (see DrumScheduler).
"""

from __future__ import annotations


class OnsetDetector:
    """Rising-edge detector on motion energy with a refractory cooldown."""

    def __init__(self, threshold: float = 0.12, refractory: int = 4, floor: float = 0.08) -> None:
        self.threshold = threshold      # min jump in energy between frames to count
        self.refractory = refractory    # frames to wait before firing again
        self.floor = floor              # ignore spikes below this absolute energy
        self._prev = 0.0
        self._cooldown = 0

    def __call__(self, energy: float) -> bool:
        onset = False
        rise = energy - self._prev
        if self._cooldown > 0:
            self._cooldown -= 1
        elif rise > self.threshold and energy > self.floor:
            onset = True
            self._cooldown = self.refractory
        self._prev = energy
        return onset
