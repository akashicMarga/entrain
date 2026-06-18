"""Selection + mixing engine. [FROZEN — the de-risked output stage]

The loop does NOT have to generate. This engine drives a real track library by the same
style vector: embed each track once, pick the nearest to the current style target, and
crossfade. Lower-risk and reliably musical — no uncanny-valley generation risk. Prove the
closed loop with THIS first, then swap in Mrt2Engine. Same MusicEngine interface.

Reuses the generator's audio encoder to embed tracks into the shared style space, so the
SAME style vector that would steer generation instead selects from the library.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from entrain.types import AudioChunk, StyleVector


class SelectionEngine:
    """Nearest-anchor track selection + crossfade over a real library."""

    def __init__(self, library_dir: str, embed_audio_fn, sample_rate: int = 48_000) -> None:
        self.library_dir = Path(library_dir)
        self.embed_audio_fn = embed_audio_fn     # frozen audio encoder: file -> (d,)
        self.sample_rate = sample_rate
        self._track_embeddings: np.ndarray | None = None   # (n_tracks, d)
        self._tracks: list[Path] = []
        self._style: StyleVector | None = None
        self._tempo = 0.0
        self._current = None
        self._embed_text_fn = None

    def index_library(self) -> None:
        """Embed every track once into the shared style space."""
        raise NotImplementedError(
            "Scan library_dir, embed each track via embed_audio_fn, cache the matrix."
        )

    def embed_text(self, prompt: str) -> np.ndarray:
        raise NotImplementedError("Use the shared text encoder for anchor prompts.")

    def start(self) -> None:
        self.index_library()

    def set_style(self, style: StyleVector) -> None:
        self._style = style

    def set_tempo(self, bpm: float) -> None:
        # Selection could beat-match via time-stretch to this target; no-op in the stub.
        self._tempo = float(bpm)

    def set_onset(self, onset: bool) -> None:
        pass                            # no drum channel for track selection

    def set_melody(self, pitch: int | None) -> None:
        pass                            # no note channel for track selection

    def next_chunk(self) -> AudioChunk:
        raise NotImplementedError(
            "Pick nearest track to self._style.vec, crossfade from current, emit a block."
        )

    def stop(self) -> None:
        self._current = None
