"""Music engine interface. [the de-risking swap point]

A `MusicEngine` is a frozen renderer steered by a style vector. Both the generative path
(mrt2) and the de-risked selection path (selection) implement this interface, so the
output stage is a one-line swap: prove the closed loop with selection first, then swap in
generation once the policy is trustworthy — same camera, same signal, same policy.

The engine also exposes the frozen text encoder used to embed anchor prompts once
(anchors.py), since text and audio share one style space (MusicCoCa-style).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from entrain.types import AudioChunk, StyleVector


class MusicEngine(Protocol):
    def embed_text(self, prompt: str) -> np.ndarray:
        """Frozen text encoder: prompt -> style embedding (d,). Used by AnchorBank."""
        ...

    def start(self) -> None:
        ...

    def set_style(self, style: StyleVector) -> None:
        """Update the steering target. Called by the vision loop, read by the audio loop."""
        ...

    def set_tempo(self, bpm: float) -> None:
        """Separate tempo control channel (NOT carried in the style vector). 0 = unknown.
        Engines that don't act on tempo may no-op."""
        ...

    def set_onset(self, onset: bool) -> None:
        """A movement accent this step -> a drum hit, for engines with a drum channel.
        No-op for engines without one."""
        ...

    def set_melody(self, pitch: int | None) -> None:
        """Lead MIDI pitch to play (None = silent), for engines with a note channel.
        No-op for engines without one."""
        ...

    def next_chunk(self) -> AudioChunk:
        """Render the next audio block. Called on the fast audio loop; must not stall."""
        ...

    def stop(self) -> None:
        ...
