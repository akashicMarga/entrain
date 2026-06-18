"""Stage 1 control policy. [NO TRAINING — hand-written rule]

The simplest thing that closes the loop: more activation -> slide weight toward
higher-energy anchors. This proves the end-to-end loop on a MacBook with zero training.
Replaced (not rewritten) by policy/learned.py in Stage 3 — same signature.

Driven by `arousal` (the estimator fuses motion + facial arousal into it, and sets it
equal to motion energy when no face is present, so this is movement-only until affect is
on). `synchrony_weight` mixes in cross-subject synchrony; `valence_weight` lets a positive
mood nudge slightly higher and a negative one lower. Anchors are ordered low -> high energy.
"""

from __future__ import annotations

import numpy as np

from entrain.types import StateVector


class HeuristicPolicy:
    """Maps arousal/synchrony/valence to a soft position along the low->high anchor axis."""

    def __init__(
        self,
        anchor_names: list[str],
        sharpness: float = 4.0,
        synchrony_weight: float = 0.0,
        valence_weight: float = 0.25,
        hr_weight: float = 0.2,
    ) -> None:
        self.anchor_names = anchor_names        # ordered low-energy -> high-energy
        self.sharpness = sharpness              # how peaked the weighting is
        # Stage 1 pose can't measure synchrony -> default 0.0 (arousal spans the axis).
        # Raise this once MediaPipe supplies real cross-subject synchrony.
        self.synchrony_weight = synchrony_weight
        self.valence_weight = valence_weight    # happy nudges up, sad down
        self.hr_weight = hr_weight              # elevated heart rate builds tension/energy

    def __call__(self, state: StateVector) -> dict[str, float]:
        k = len(self.anchor_names)
        # One "drive" from all signals: movement (arousal) is primary, mood and heart
        # tension push it up/down. Then optional crowd-synchrony mixing.
        drive = state.arousal
        drive += self.valence_weight * state.valence
        drive += self.hr_weight * state.hr_tension
        sw = self.synchrony_weight
        pos = (1 - sw) * drive + sw * state.synchrony
        pos = float(np.clip(pos, 0.0, 1.0))
        centers = np.linspace(0.0, 1.0, k)
        # Soft assignment: closer anchors get more weight.
        logits = -self.sharpness * (centers - pos) ** 2
        w = np.exp(logits - logits.max())
        w = w / w.sum()
        return dict(zip(self.anchor_names, w.tolist()))
