"""Control policy interface. [the seam between heuristic and learned]

A policy maps a `StateVector` to anchor weights. The heuristic (Stage 1) and the learned
MLX net (Stage 3) implement the SAME signature, so swapping them is one line in app.py.
The output type never changes — always weights, hence (after blending) a style vector.
"""

from __future__ import annotations

from typing import Protocol

from entrain.types import StateVector


class Policy(Protocol):
    """state -> anchor weights. Implemented by HeuristicPolicy and LearnedPolicy."""

    def __call__(self, state: StateVector) -> dict[str, float]:
        ...
