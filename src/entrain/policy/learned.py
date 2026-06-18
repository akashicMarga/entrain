"""Stage 3 control policy. [🪸 TRAINED — coral box #2, THE contribution]

The original problem. Maps crowd state -> musical move to maximize engagement/SYNCHRONY.
This is closed-loop, not supervised: the right action isn't in any dataset — it's learned
from how the crowd responds AFTER you act. Trained offline (RL / imitation from festival
footage) to avoid live experimentation. Reward = cross-crowd synchrony, not average arousal.

Must be shaped for RESTRAINT and narrative arc — a naive reactive controller oscillates,
lags, or always-ramps-up. A human DJ has restraint; a controller has none unless designed in.

Training entrypoint: training/policy/train_policy.py
Same signature as HeuristicPolicy, so app.py swaps one for the other.
"""

from __future__ import annotations

from entrain.types import StateVector


class LearnedPolicy:
    """MLX net: StateVector -> anchor weights. Drop-in for HeuristicPolicy."""

    def __init__(self, anchor_names: list[str], weights_path: str | None = None) -> None:
        self.anchor_names = anchor_names
        self.weights_path = weights_path
        self._net = None      # 🪸 MLX module trained offline

    def load(self) -> None:
        raise NotImplementedError(
            "Load trained MLX policy net (Stage 3). Output dim = len(anchor_names)."
        )

    def __call__(self, state: StateVector) -> dict[str, float]:
        raise NotImplementedError
        # logits = self._net(state.as_array())          # (k,)
        # w = softmax(logits)
        # return dict(zip(self.anchor_names, w.tolist()))
