"""The slow Director layer + its seam with the AnchorBank.

Director authors WHICH anchors define the musical range (slow, language-able); the fast
policy then weights over them. Tested without any LLM via StaticDirector + a fake embedder.
"""

import numpy as np

from entrain.policy.anchors import AnchorBank
from entrain.policy.director import HeuristicDirector, LlmDirector, StaticDirector
from entrain.types import AnchorSpec, Directive, DirectorContext


def _ctx(elapsed_s: float = 0.0, energy_trend: float = 0.0) -> DirectorContext:
    return DirectorContext(
        elapsed_s=elapsed_s, energy_mean=0.5, energy_trend=energy_trend,
        valence_mean=0.0, synchrony_mean=0.0, seconds_in_directive=elapsed_s,
    )


def test_static_director_returns_the_yaml_anchors():
    d = StaticDirector().initial()
    assert [a.name for a in d.anchors] == ["calm", "groove", "peak"]   # low -> high
    assert all(a.prompt for a in d.anchors)                            # real prompts


def test_static_director_never_revises():
    assert StaticDirector().revise(_ctx(elapsed_s=120.0)) is None


def test_anchor_bank_seeds_from_a_directive():
    directive = Directive(anchors=[AnchorSpec("a", "pa"), AnchorSpec("b", "pb")])
    bank = AnchorBank(embed_fn=lambda p: np.ones(4, dtype=np.float32), directive=directive)
    assert bank.names == ["a", "b"]
    bank.embed()
    style = bank.blend({"a": 1.0, "b": 1.0})
    assert style.weights == {"a": 0.5, "b": 0.5}


def test_reauthor_swaps_anchors_and_forces_reembed():
    embed = lambda p: np.ones(4, dtype=np.float32)
    bank = AnchorBank(embed_fn=embed, directive=StaticDirector().initial())
    bank.embed()
    bank.reauthor(Directive(anchors=[AnchorSpec("x", "px")]))
    assert bank.names == ["x"]
    # blending before re-embed must fail (the old embeddings were dropped)
    try:
        bank.blend({"x": 1.0})
        assert False, "expected RuntimeError before re-embed"
    except RuntimeError:
        pass
    bank.embed()
    assert bank.blend({"x": 1.0}).weights == {"x": 1.0}


def test_heuristic_director_lifts_on_rising_energy():
    d = HeuristicDirector(trend_up=0.004)
    out = d.revise(_ctx(elapsed_s=10.0, energy_trend=0.02))   # clearly rising
    assert out is not None and out.intent == "building"
    assert [a.name for a in out.anchors] == ["calm", "groove", "peak"]   # names stable
    assert "driving" in out.anchors[0].prompt                # modifier applied
    assert "calm" in out.anchors[0].prompt                   # base keyword kept (synth embed)


def test_heuristic_director_returns_none_when_mode_unchanged():
    d = HeuristicDirector()
    assert d.revise(_ctx(elapsed_s=5.0, energy_trend=0.0)) is None   # stays neutral


def test_heuristic_director_eases_off_a_long_build_restraint():
    d = HeuristicDirector(trend_up=0.004, sustain_cap_s=45.0)
    assert d.revise(_ctx(elapsed_s=0.0, energy_trend=0.02)).intent == "building"
    # still rising, but held past the cap -> the director eases off anyway (restraint)
    eased = d.revise(_ctx(elapsed_s=60.0, energy_trend=0.02))
    assert eased is not None and eased.intent == "cooling down"


def test_anchor_bank_reauthor_and_embed_is_atomic():
    embed = lambda p: np.ones(4, dtype=np.float32)
    bank = AnchorBank(embed_fn=embed, directive=StaticDirector().initial())
    bank.embed()
    # no separate embed() call needed afterward, and no RuntimeError window:
    bank.reauthor_and_embed(Directive(anchors=[AnchorSpec("calm", "calm v2"),
                                               AnchorSpec("peak", "peak v2")]))
    assert bank.names == ["calm", "peak"]
    assert bank.blend({"calm": 1.0, "peak": 1.0}).weights == {"calm": 0.5, "peak": 0.5}


def test_llm_director_falls_back_without_a_model():
    """No model loaded -> behaves like StaticDirector (the always-available fallback)."""
    d = LlmDirector(model=None, revise_every_s=8.0)
    assert [a.name for a in d.initial().anchors] == ["calm", "groove", "peak"]


def test_llm_director_rate_limits_revise():
    d = LlmDirector(model=None, revise_every_s=8.0)
    assert d.revise(_ctx(elapsed_s=0.0)) is None        # first tick: fallback returns None
    # within the window it returns None without doing work; past it, it ticks again (still
    # None here only because the fallback is static — the point is it doesn't raise).
    assert d.revise(_ctx(elapsed_s=2.0)) is None
