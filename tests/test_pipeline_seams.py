"""Tests for the seams that have real logic in Stage 1.

The frozen/heavy boxes (camera, pose, affect, rppg, engines) are stubs that raise
NotImplementedError. What IS implemented and worth testing: anchor blending, the
heuristic policy axis, state smoothing, and the rate-limited style slot.
"""

import numpy as np

from entrain.policy.anchors import AnchorBank
from entrain.policy.heuristic import HeuristicPolicy
from entrain.runtime.shared_state import StyleSlot
from entrain.state.estimator import StateEstimator
from entrain.types import (
    AffectFeatures,
    PoseFeatures,
    StateVector,
    StyleVector,
)

ANCHOR_NAMES = ["calm", "groove", "peak"]


def _fake_bank():
    # 3 orthonormal anchors in 4-d so blends are easy to reason about.
    embs = np.eye(3, 4, dtype=np.float32)
    bank = AnchorBank.__new__(AnchorBank)
    from entrain.types import AnchorSet

    bank._anchors = AnchorSet(names=ANCHOR_NAMES, embeddings=embs)
    return bank


def test_anchor_blend_is_normalized_weighted_sum():
    bank = _fake_bank()
    style = bank.blend({"calm": 1.0, "groove": 3.0, "peak": 0.0})
    # weights normalize to 0.25 / 0.75 / 0.0
    assert style.weights["calm"] == 0.25
    assert style.weights["groove"] == 0.75
    np.testing.assert_allclose(style.vec, [0.25, 0.75, 0.0, 0.0], atol=1e-6)


def test_heuristic_moves_toward_peak_with_arousal():
    pol = HeuristicPolicy(anchor_names=ANCHOR_NAMES)
    low = pol(StateVector(arousal=0.0, valence=0, energy=0.0, synchrony=0.0, tempo=120))
    high = pol(StateVector(arousal=1.0, valence=0, energy=1.0, synchrony=1.0, tempo=120))
    assert low["calm"] > low["peak"]        # quiet room -> calm anchor
    assert high["peak"] > high["calm"]      # activated room -> peak anchor


def test_heuristic_valence_weight_nudges_position():
    pol = HeuristicPolicy(anchor_names=ANCHOR_NAMES, valence_weight=0.3)
    base = StateVector(arousal=0.5, valence=0.0, energy=0.5, synchrony=0.0, tempo=0)
    happy = StateVector(arousal=0.5, valence=1.0, energy=0.5, synchrony=0.0, tempo=0)
    assert pol(happy)["peak"] > pol(base)["peak"]   # positive mood -> nudged higher


def test_state_estimator_smooths():
    est = StateEstimator(attack=0.5, release=0.5)
    p0 = PoseFeatures(motion_energy=0.0, movement_tempo=120, synchrony=0.0)
    p1 = PoseFeatures(motion_energy=1.0, movement_tempo=120, synchrony=1.0)
    est.update(p0)
    s1 = est.update(p1)
    # second update should land partway, not jump straight to 1.0
    assert 0.0 < s1.energy < 1.0


def test_style_slot_rate_limits():
    slot = StyleSlot(max_step=0.1)
    slot.write(StyleVector(vec=np.zeros(4, dtype=np.float32)))
    slot.write(StyleVector(vec=np.array([1, 0, 0, 0], dtype=np.float32)))
    moved = float(np.linalg.norm(slot.read().vec))
    assert moved <= 0.1 + 1e-6            # one big jump is clamped to max_step


def test_state_estimator_holds_the_vibe():
    """Fast to rise, slow to fall: after a single still frame, energy stays high."""
    est = StateEstimator(attack=0.5, release=0.04)
    moving = PoseFeatures(motion_energy=1.0, movement_tempo=0, synchrony=0.0)
    still = PoseFeatures(motion_energy=0.0, movement_tempo=0, synchrony=0.0)
    for _ in range(8):
        est.update(moving)              # ramp up
    risen = est.update(moving).energy
    after_one_still = est.update(still).energy
    # one still frame barely dents it (release is slow) ...
    assert after_one_still > 0.9 * risen
    # ... but sustained stillness does eventually bring it down.
    for _ in range(60):
        est.update(still)
    assert est.update(still).energy < 0.2


def test_state_estimator_uses_affect_when_present():
    est = StateEstimator(attack=1.0, release=1.0)  # no smoothing, read raw fusion
    pose = PoseFeatures(motion_energy=0.0, movement_tempo=120, synchrony=0.0)
    affect = AffectFeatures(valence=0.8, arousal=1.0, n_faces=3)
    s = est.update(pose, affect)
    assert s.valence == 0.8
    assert s.arousal == 0.5  # 0.5*pose(0) + 0.5*affect(1)


def test_micromotion_drives_energy_when_gross_motion_is_zero():
    """A seated, still-but-engaged listener (motion_energy 0) still steers via micromotion."""
    est = StateEstimator(attack=1.0, release=1.0, micro_weight=0.6)  # raw, no smoothing
    still_dead = PoseFeatures(motion_energy=0.0, movement_tempo=0, synchrony=0.0,
                              micromotion=0.0)
    still_engaged = PoseFeatures(motion_energy=0.0, movement_tempo=0, synchrony=0.0,
                                 micromotion=1.0)
    assert est.update(still_dead).energy == 0.0
    # micro_weight * micromotion (tolerance: state vector is stored float32)
    assert abs(est.update(still_engaged).energy - 0.6) < 1e-6


def test_gross_motion_wins_over_micromotion():
    """A dancer isn't penalized: gross motion dominates the weaker micro contribution."""
    est = StateEstimator(attack=1.0, release=1.0, micro_weight=0.6)
    dancing = PoseFeatures(motion_energy=0.9, movement_tempo=120, synchrony=0.0,
                           micromotion=1.0)
    assert est.update(dancing).energy == 0.9            # max(0.9, 0.6*1.0)


def test_style_slot_freezes_when_not_responsive():
    """Static-generative baseline: first style writes through, later ones are ignored."""
    slot = StyleSlot(max_step=1.0, responsive=False)
    slot.write(StyleVector(vec=np.array([1, 0, 0, 0], dtype=np.float32)), tempo=120)
    slot.write(StyleVector(vec=np.array([0, 1, 0, 0], dtype=np.float32)), tempo=60)
    np.testing.assert_allclose(slot.read().vec, [1, 0, 0, 0])   # held at the first style
    assert slot.tempo == 120                                    # tempo frozen too
    # movement accents are suppressed in the frozen baseline
    slot.set_onset()
    slot.set_melody(60)
    assert slot.take_onset() is False
    assert slot.melody is None


def test_style_slot_responsive_still_updates():
    slot = StyleSlot(max_step=2.0, responsive=True)   # > sqrt(2) jump, no rate-limit here
    slot.write(StyleVector(vec=np.array([1, 0, 0, 0], dtype=np.float32)))
    slot.write(StyleVector(vec=np.array([0, 1, 0, 0], dtype=np.float32)))
    np.testing.assert_allclose(slot.read().vec, [0, 1, 0, 0])   # steering still moves
