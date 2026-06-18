"""Stage 3 — train the control policy. [🪸 coral box #2, THE contribution]

The original research. The right action isn't in any dataset, so this is closed-loop, not
supervised. Learn it OFFLINE (RL / imitation from festival footage: music paired with
crowd reaction) to avoid live experimentation.

Reward = cross-crowd SYNCHRONY (inter-subject correlation over ~10s windows), NOT average
arousal — a loud bad drop spikes arousal but not synchrony. Shape for restraint and
narrative arc so the loop doesn't oscillate or always-ramp-up.

Output: weights consumed by policy/learned.py (LearnedPolicy._net).
"""

from __future__ import annotations


def synchrony_reward(*args, **kwargs):
    """Inter-subject correlation of movement/physiology over a window. The reward signal."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError(
        "Stage 3: build (state, action, next-state, synchrony-reward) transitions from "
        "festival footage -> offline RL / imitation -> MLX policy net -> evaluate against "
        "the Stage-1 heuristic on held-out engagement/synchrony."
    )


if __name__ == "__main__":
    main()
