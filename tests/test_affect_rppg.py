"""Affect blendshape->VA mapping and rPPG POS heart-rate DSP, tested without a camera."""

import numpy as np

from entrain.perception.affect import MediaPipeAffectReader
from entrain.perception.rppg import pos_heart_rate
from entrain.state.estimator import StateEstimator
from entrain.types import AffectFeatures, CardiacFeatures, PoseFeatures


# --- affect: blendshapes -> valence / arousal -----------------------------------

def test_smile_is_positive_valence():
    va = MediaPipeAffectReader._blendshapes_to_va(
        {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8}
    )
    assert va[0] > 0.3


def test_frown_is_negative_valence():
    va = MediaPipeAffectReader._blendshapes_to_va(
        {"mouthFrownLeft": 0.7, "mouthFrownRight": 0.7}
    )
    assert va[0] < -0.3


def test_open_mouth_wide_eyes_is_high_arousal():
    calm = MediaPipeAffectReader._blendshapes_to_va({})
    excited = MediaPipeAffectReader._blendshapes_to_va(
        {"jawOpen": 0.9, "eyeWideLeft": 0.8, "eyeWideRight": 0.8, "browInnerUp": 0.6}
    )
    assert excited[1] > calm[1]
    assert excited[1] > 0.4


def test_neutral_face_is_near_zero():
    v, a = MediaPipeAffectReader._blendshapes_to_va({})
    assert v == 0.0 and a == 0.0


# --- rPPG: POS heart rate -------------------------------------------------------

def test_pos_recovers_known_heart_rate():
    fps, hr_bpm = 30, 72.0
    n = int(fps * 8)
    t = np.arange(n) / fps
    pulse = np.sin(2 * np.pi * (hr_bpm / 60.0) * t)
    # green channel carries the strongest plethysmographic signal
    rgb = np.stack([
        120 + 0.3 * pulse,
        130 + 1.0 * pulse,
        110 + 0.5 * pulse,
    ], axis=1) + np.random.default_rng(0).normal(0, 0.05, (n, 3))
    bpm, conf = pos_heart_rate(rgb, fps)
    assert abs(bpm - hr_bpm) < 6
    assert conf > 0.1


def test_pos_too_few_samples_returns_nan():
    bpm, conf = pos_heart_rate(np.zeros((4, 3)), 30)
    assert np.isnan(bpm) and conf == 0.0


# --- estimator: affect + cardiac integration ------------------------------------

def test_estimator_fuses_affect_and_carries_heart_rate():
    est = StateEstimator(attack=1.0, release=1.0)   # no smoothing -> read raw fusion
    pose = PoseFeatures(motion_energy=0.0, movement_tempo=0, synchrony=0.0)
    affect = AffectFeatures(valence=0.6, arousal=1.0, n_faces=1)
    cardiac = CardiacFeatures(heart_rate=68.0, hr_trend=1.0, confidence=0.4)
    s = est.update(pose, affect, cardiac)
    assert s.valence == 0.6
    assert s.arousal == 0.5            # 0.5*pose(0) + 0.5*face(1)
    assert s.heart_rate == 68.0        # carried alongside, not in as_array

    # low-confidence rPPG: heart_rate holds its last value instead of flickering
    s2 = est.update(pose, affect, CardiacFeatures(heart_rate=float("nan"), hr_trend=0, confidence=0.0))
    assert s2.heart_rate == 68.0
