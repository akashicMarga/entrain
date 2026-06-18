"""Tempo + synchrony DSP (the MediaPipe reader's analysis half), tested with synthetic
bob signals so it needs no camera or model."""

import numpy as np

from entrain.perception.pose import _MotionTracker


def _feed(tracker, pose_idx, signal):
    for v in signal:
        tracker.push(pose_idx, float(v))


def test_tempo_recovers_known_rhythm():
    fps = 30
    tracker = _MotionTracker(fps=fps, history_s=6.0)
    t = np.arange(int(fps * 6)) / fps
    # 2 Hz bob == 120 movement BPM
    _feed(tracker, 0, 0.5 + 0.1 * np.sin(2 * np.pi * 2.0 * t))
    assert abs(tracker.tempo_bpm(0) - 120.0) < 10.0


def test_tempo_zero_when_still():
    tracker = _MotionTracker(fps=30, history_s=6.0)
    _feed(tracker, 0, np.full(200, 0.5))      # no movement
    assert tracker.tempo_bpm(0) == 0.0


def test_tempo_zero_before_enough_history():
    tracker = _MotionTracker(fps=30, history_s=6.0)
    _feed(tracker, 0, np.random.rand(10))     # far short of the window
    assert tracker.tempo_bpm(0) == 0.0


def test_synchrony_high_for_in_phase_dancers():
    fps = 30
    tracker = _MotionTracker(fps=fps, history_s=6.0, sync_window_s=4.0)
    t = np.arange(int(fps * 6)) / fps
    base = np.sin(2 * np.pi * 1.5 * t)
    _feed(tracker, 0, 0.5 + 0.1 * base)
    _feed(tracker, 1, 0.5 + 0.1 * base)       # same motion -> locked
    assert tracker.synchrony() > 0.9


def test_synchrony_low_for_anti_phase_dancers():
    fps = 30
    tracker = _MotionTracker(fps=fps, history_s=6.0, sync_window_s=4.0)
    t = np.arange(int(fps * 6)) / fps
    base = np.sin(2 * np.pi * 1.5 * t)
    _feed(tracker, 0, 0.5 + 0.1 * base)
    _feed(tracker, 1, 0.5 - 0.1 * base)       # anti-correlated -> not synchrony
    assert tracker.synchrony() == 0.0


def test_synchrony_zero_with_one_person():
    tracker = _MotionTracker(fps=30, history_s=6.0, sync_window_s=4.0)
    _feed(tracker, 0, np.random.rand(200))    # only pose 0 ever seen
    assert tracker.synchrony() == 0.0


# --- micromotion: the near-still listener's subtle-movement intensity ---------------

def test_micromotion_zero_when_perfectly_still():
    tracker = _MotionTracker(fps=30, micro_window_s=1.5, micro_gain=100.0)
    _feed(tracker, 0, np.full(200, 0.5))      # frozen -> no fluctuation
    assert tracker.micromotion(0) == 0.0


def test_micromotion_zero_before_window_fills():
    tracker = _MotionTracker(fps=30, micro_window_s=1.5, micro_gain=100.0)
    _feed(tracker, 0, np.random.rand(20))     # 20 < the 45-sample micro window
    assert tracker.micromotion(0) == 0.0


def test_micromotion_positive_for_subtle_sway():
    fps = 30
    tracker = _MotionTracker(fps=fps, micro_window_s=1.5, micro_gain=100.0)
    t = np.arange(int(fps * 2)) / fps
    # a tiny 0.01-amplitude sway -> std ~0.0071 -> *100 ~ 0.71 (well below the clip)
    _feed(tracker, 0, 0.5 + 0.01 * np.sin(2 * np.pi * 1.5 * t))
    micro = tracker.micromotion(0)
    assert 0.3 < micro < 1.0


def test_micromotion_rises_with_amplitude_then_clips():
    fps = 30
    t = np.arange(int(fps * 2)) / fps

    def micro_for(amp: float) -> float:
        tr = _MotionTracker(fps=fps, micro_window_s=1.5, micro_gain=100.0)
        _feed(tr, 0, 0.5 + amp * np.sin(2 * np.pi * 1.5 * t))
        return tr.micromotion(0)

    assert micro_for(0.005) < micro_for(0.01)   # monotonic below the clip
    assert micro_for(0.5) == 1.0                # gross movement saturates to 1.0


def test_micromotion_drops_static_posture():
    """A held-off-center posture is still (zero fluctuation) -> not movement."""
    tracker = _MotionTracker(fps=30, micro_window_s=1.5, micro_gain=100.0)
    _feed(tracker, 0, np.full(200, 0.3))      # parked off-center, but frozen
    assert tracker.micromotion(0) == 0.0
