"""Anchor prompt management. [BUILD — embeds frozen prompts once]

Predefine a few anchor prompts spanning the musical range, embed each ONCE into the
generator's shared text/audio style space (MusicCoCa-style), and cache the result. The
control policy then expresses crowd state as weights over these anchors; the style vector
is the weighted blend.

The anchors are config (configs/anchors.yaml), not code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from entrain.types import AnchorSet, Directive, StyleVector


class AnchorBank:
    """Loads anchor prompts, embeds them once via the generator's text encoder.

    Prompts come either from `configs/anchors.yaml` (Stage-1 default) or from a Director's
    `Directive` — the slow planning layer authors them. Either way the names are read
    eagerly (the policy needs them at construction); the actual EMBEDDING is deferred to
    `embed()` because for a generative engine (MRT2) it must run on the audio thread.
    """

    def __init__(self, embed_fn, config_path: str = "configs/anchors.yaml",
                 directive: Directive | None = None) -> None:
        # embed_fn: str -> np.ndarray (d,)  — the generator's frozen text encoder.
        self.embed_fn = embed_fn
        self.config_path = Path(config_path)
        self._anchors: AnchorSet | None = None
        if directive is not None:
            self._set_prompts(directive)
        else:
            import yaml

            self._prompts = yaml.safe_load(self.config_path.read_text())["anchors"]
            self._names = [p["name"] for p in self._prompts]

    def _set_prompts(self, directive: Directive) -> None:
        self._prompts = [{"name": a.name, "prompt": a.prompt} for a in directive.anchors]
        self._names = [a.name for a in directive.anchors]

    def reauthor(self, directive: Directive) -> None:
        """Swap in a new Director-authored anchor set. The caller must call `embed()`
        afterward — on the generation thread, like the initial warmup — before blending."""
        self._set_prompts(directive)
        self._anchors = None        # force a re-embed; blend() raises until embed() runs

    @property
    def names(self) -> list[str]:
        return self._names

    def embed(self) -> AnchorSet:
        """Embed each anchor prompt once. Call on the engine's generation thread."""
        embeddings = np.stack([self.embed_fn(p["prompt"]) for p in self._prompts])
        self._anchors = AnchorSet(names=self._names, embeddings=embeddings)
        return self._anchors

    # Backwards-compatible alias.
    load = embed

    @property
    def anchors(self) -> AnchorSet:
        if self._anchors is None:
            raise RuntimeError("Call embed() first.")
        return self._anchors

    def blend(self, weights: dict[str, float]) -> StyleVector:
        """Σ weightᵢ · anchorᵢ -> the ~768-d signal on the wire."""
        a = self.anchors
        w = np.array([weights.get(n, 0.0) for n in a.names], dtype=np.float32)
        s = w.sum()
        if s > 0:
            w = w / s
        vec = (w[:, None] * a.embeddings).sum(axis=0)
        return StyleVector(vec=vec, weights=dict(zip(a.names, w.tolist())))
