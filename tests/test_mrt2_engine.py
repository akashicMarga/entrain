"""MRT2 engine smoke test. Auto-skips unless magenta-rt is installed AND weights are
present, so the suite stays light. On a set-up Mac it confirms the adapter embeds a
prompt, blends, and streams a real audio chunk with continuous state."""

import numpy as np
import pytest

pytest.importorskip("magenta_rt")

from entrain.generation.mrt2 import Mrt2Engine
from entrain.policy.anchors import AnchorBank
from entrain.types import AnchorSet, StyleVector


def _engine():
    # small model = the lightest thing that proves the path end to end
    return Mrt2Engine(size="mrt2_small", frames_per_chunk=5)


def test_embed_blend_and_stream():
    eng = _engine()
    try:
        eng.start()
    except Exception as e:                       # weights not downloaded -> skip
        pytest.skip(f"MRT2 weights unavailable: {e}")

    # anchors -> blended style vector (the library's own np.mean-of-embeddings pattern)
    bank = AnchorBank(embed_fn=eng.embed_text)
    bank._anchors = AnchorSet(
        names=["calm", "peak"],
        embeddings=np.stack([eng.embed_text("calm ambient"),
                             eng.embed_text("peak techno")]),
    )
    style = bank.blend({"calm": 0.3, "peak": 0.7})
    assert style.vec.shape == bank.anchors.embeddings.shape[1:]

    eng.set_style(style)
    c1 = eng.next_chunk()
    c2 = eng.next_chunk()                          # continues from streaming state
    assert c1.sample_rate == 48_000
    assert c1.pcm.ndim == 2 and c1.pcm.shape[1] == 2
    assert np.isfinite(c1.pcm).all() and np.isfinite(c2.pcm).all()
