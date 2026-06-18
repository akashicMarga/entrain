"""Melody mapping + note scheduling, and affect/heart folding into the steering drive."""

import numpy as np

from entrain.generation.mrt2 import NoteScheduler
from entrain.perception.melody import MelodyMapper
from entrain.policy.heuristic import HeuristicPolicy
from entrain.state.estimator import StateEstimator
from entrain.types import AffectFeatures, CardiacFeatures, PoseFeatures, StateVector

NAMES = ["calm", "groove", "peak"]


# --- melody ---------------------------------------------------------------------

def test_hand_down_is_silent():
    mm = MelodyMapper(active_threshold=0.35)
    assert mm(0.0) is None
    assert mm(0.2) is None


def test_higher_hand_higher_pitch_and_in_scale():
    mm = MelodyMapper(root=48, octaves=2, active_threshold=0.3)
    low = mm(0.35)
    high = mm(0.98)
    assert low is not None and high is not None
    assert high > low                                  # raise hand -> go up
    pentatonic = {0, 2, 4, 7, 9}
    for p in (low, high):
        assert (p - 48) % 12 in pentatonic              # stays on the scale


def test_note_scheduler_onset_then_sustain():
    ns = NoteScheduler(enabled=True)
    n1 = ns.step(60)
    assert n1[60] == 2 and n1.count(-1) == 127          # new pitch = onset, rest masked
    n2 = ns.step(60)
    assert n2[60] == 1                                  # same pitch held = sustain
    n3 = ns.step(None)
    assert all(v == -1 for v in n3)                     # released = all masked


def test_note_scheduler_disabled():
    assert NoteScheduler(enabled=False).step(60) is None


# --- affect / heart steering ----------------------------------------------------

def test_valence_and_hr_push_the_drive_up():
    pol = HeuristicPolicy(NAMES, valence_weight=0.3, hr_weight=0.3)
    base = StateVector(arousal=0.4, valence=0.0, energy=0.4, synchrony=0, tempo=0)
    happy_excited = StateVector(arousal=0.4, valence=1.0, energy=0.4, synchrony=0,
                                tempo=0, hr_tension=1.0)
    assert pol(happy_excited)["peak"] > pol(base)["peak"]


def test_estimator_builds_hr_tension_above_baseline():
    est = StateEstimator(attack=1.0, release=1.0, hr_range=12.0)
    pose = PoseFeatures(motion_energy=0.0, movement_tempo=0, synchrony=0.0)
    # establish a resting baseline ~60
    for _ in range(50):
        est.update(pose, cardiac=CardiacFeatures(heart_rate=60.0, hr_trend=0, confidence=0.5))
    rest = est.update(pose, cardiac=CardiacFeatures(heart_rate=60.0, hr_trend=0, confidence=0.5))
    spiked = est.update(pose, cardiac=CardiacFeatures(heart_rate=75.0, hr_trend=5, confidence=0.5))
    assert abs(rest.hr_tension) < 0.2          # at baseline -> ~0
    assert spiked.hr_tension > 0.5             # well above baseline -> tension
