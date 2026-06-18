"""Stage 2 — train the affect read-out. [🪸 coral box #1]

Conventional, modest supervised learning. Adapt a small MLX head on a pretrained FER
backbone to real crowd conditions using collected/labelled festival footage, producing
valence/arousal. The frozen face detector is NOT trained; only the head.

Output: weights consumed by perception/affect.py (AffectReader._head).
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Stage 2: load festival FER dataset -> MLX head on frozen backbone -> "
        "regress valence/arousal -> save weights for AffectReader."
    )


if __name__ == "__main__":
    main()
