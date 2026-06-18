"""Per-session signal calibration. [BUILD — no training, but the input a learned one wants]

A fixed threshold like ``onset_threshold: 0.12`` is wrong for the next person: one
listener's idle fidget is another's full effort, and every camera/posture has its own
baseline jitter. This calibrates a movement signal to the person in front of it.

It tracks a RESTING FLOOR — an asymmetric EMA that falls fast (quickly finds your
resting/quiet level) and rises slowly (so sustained dancing doesn't get normalized away
into "rest") — plus the typical SPREAD of movement above that floor. Downstream gates
then ask "is this real movement above your own rest?" (``excess``) or "how big a jump
relative to your usual fluctuation?" (``excess / spread``) instead of trusting a constant.

Same idea as the state estimator's ``hr_baseline`` -> ``hr_tension``, generalized. It also
produces exactly the normalized representation a learned controller (``policy/learned.py``)
would consume — built here so the heuristic gates and the future learned policy share one
calibrated input.
"""

from __future__ import annotations


class SignalCalibrator:
    """Tracks a resting floor + spread of one signal; reports movement above rest."""

    def __init__(
        self,
        fall: float = 0.05,        # floor tracks DOWN fast -> finds your resting level
        rise: float = 0.0015,      # ... and UP slowly -> sustained motion stays "above rest"
        spread_alpha: float = 0.01,
        spread_floor: float = 0.02,  # min spread (avoids over-sensitivity when very still)
    ) -> None:
        self.fall = fall
        self.rise = rise
        self.spread_alpha = spread_alpha
        self.spread_floor = spread_floor
        self._floor: float | None = None
        self._spread = spread_floor

    def update(self, x: float) -> None:
        """Push one sample, advancing the resting floor and the spread estimate."""
        x = float(x)
        if self._floor is None:
            self._floor = x
            return
        a = self.fall if x < self._floor else self.rise
        self._floor += a * (x - self._floor)
        excess = max(0.0, x - self._floor)
        self._spread += self.spread_alpha * (excess - self._spread)

    def excess(self, x: float) -> float:
        """Movement above the resting floor, in the signal's native units (0 at/below rest)."""
        if self._floor is None:
            return 0.0
        return max(0.0, float(x) - self._floor)

    @property
    def spread(self) -> float:
        return max(self._spread, self.spread_floor)

    @property
    def floor(self) -> float:
        return self._floor if self._floor is not None else 0.0
