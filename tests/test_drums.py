"""Onset detection + drum scheduling (the movement->drums logic), camera-free."""

import numpy as np

from entrain.generation.mrt2 import DrumScheduler
from entrain.perception.onset import OnsetDetector


# --- onset detector -------------------------------------------------------------

def test_onset_fires_on_a_spike_once():
    det = OnsetDetector(threshold=0.2, refractory=3, floor=0.05)
    energies = [0.0, 0.0, 0.9, 0.9, 0.9, 0.0]   # one sharp jump, then sustained
    fired = [det(e) for e in energies]
    assert fired == [False, False, True, False, False, False]  # fires once, refractory holds


def test_onset_ignores_slow_drift_and_low_energy():
    det = OnsetDetector(threshold=0.2, refractory=2, floor=0.1)
    # gradual ramp (each step below threshold) -> no onset
    assert not any(det(e) for e in np.linspace(0, 1, 20))
    det2 = OnsetDetector(threshold=0.1, refractory=2, floor=0.3)
    assert not det2(0.2)                          # jump but below the energy floor


# --- drum scheduler -------------------------------------------------------------

def test_pulse_rate_matches_tempo():
    # 120 bpm -> 2 beats/sec. 160ms chunks over ~5s.
    sch = DrumScheduler(chunk_dur=0.16, enabled=True)
    beats = sum(1 for _ in range(int(5 / 0.16)) if sch.step(120.0) == 1)
    assert abs(beats - 10) <= 1                   # ~2 beats/sec * 5s = 10


def test_onset_adds_a_hit_off_the_beat():
    sch = DrumScheduler(chunk_dur=0.16, enabled=True)
    sch.step(120.0)                               # consume one step (not yet a beat)
    sch.trigger_onset()
    assert sch.step(120.0) == 1                   # accent forces a hit even between beats


def test_masked_when_no_groove_no_accent():
    sch = DrumScheduler(chunk_dur=0.16, enabled=True)
    assert sch.step(0.0) == -1                    # no tempo, no onset -> hand back to model


def test_disabled_always_masked():
    sch = DrumScheduler(chunk_dur=0.16, enabled=False)
    sch.trigger_onset()
    assert sch.step(120.0) == -1
