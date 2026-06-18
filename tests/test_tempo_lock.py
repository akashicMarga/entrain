"""Tempo travels as a separate control channel and the synth pulse locks to it."""

import numpy as np

from entrain.generation.synth import SynthEngine
from entrain.runtime.shared_state import StyleSlot
from entrain.types import StyleVector


def _envelope_peak_hz(pcm: np.ndarray, sr: int) -> float:
    """Dominant frequency of the amplitude envelope (the pulse rate)."""
    env = np.abs(pcm[:, 0])
    env = env - env.mean()
    mag = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    freqs = np.fft.rfftfreq(len(env), d=1.0 / sr)
    band = (freqs > 0.3) & (freqs < 8.0)
    return float(freqs[band][int(np.argmax(mag[band]))])


def test_pulse_locks_to_tempo():
    sr = 48_000
    eng = SynthEngine(sample_rate=sr, block=sr)     # 1 s blocks
    eng.start()
    eng.set_style(StyleVector(vec=np.array([0, 1, 0], dtype=np.float32)))  # groove
    eng.set_tempo(120.0)                            # 120 BPM -> 2 Hz pulse
    pcm = np.concatenate([eng.next_chunk().pcm for _ in range(3)], axis=0)
    assert abs(_envelope_peak_hz(pcm, sr) - 2.0) < 0.25


def test_tempo_zero_falls_back_to_density_rate():
    eng = SynthEngine(block=1024)
    eng.start()
    eng.set_style(StyleVector(vec=np.array([0, 0, 1], dtype=np.float32)))
    eng.set_tempo(0.0)                              # unknown -> density-derived rate
    # must still produce valid audio, no crash, no NaNs
    pcm = eng.next_chunk().pcm
    assert np.isfinite(pcm).all()


def test_slot_latches_drum_hits():
    slot = StyleSlot()
    assert slot.take_drum_hit() is False        # nothing yet
    slot.set_drum(0); slot.set_drum(-1)
    assert slot.take_drum_hit() is False        # non-hits don't latch
    slot.set_drum(1)
    assert slot.take_drum_hit() is True         # a hit latches...
    assert slot.take_drum_hit() is False        # ...and is cleared on read


def test_slot_smooths_tempo_and_ignores_garbage():
    slot = StyleSlot(tempo_alpha=0.5)
    style = StyleVector(vec=np.array([1, 0, 0], dtype=np.float32))
    slot.write(style, tempo=0.0)                    # <1 BPM ignored (no rhythm yet)
    assert slot.tempo == 0.0
    slot.write(style, tempo=120.0)                  # first real reading adopted directly
    assert slot.tempo == 120.0
    slot.write(style, tempo=124.0)                  # then EMA-smoothed, not jumpy
    assert 120.0 < slot.tempo < 124.0
