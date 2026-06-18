"""Synth-tone engine. [FROZEN renderer — Stage 1 throwaway, no model, no library]

The cheapest possible output stage: a small subtractive/additive synth driven directly by
the style vector. Its only job is to make the closed loop AUDIBLE today — move in front of
the camera, the tone shifts — so the perception+policy half can be developed against real
sound before any music model or track library exists. Swap it for SelectionEngine or
Mrt2Engine later (same MusicEngine interface); nothing upstream changes.

Style space = the anchor-weight simplex. `embed_text` maps each anchor prompt to a one-hot
basis vector, so the blended style vector IS the weight over [calm, groove, peak]. Each
anchor carries a timbre preset; the live params are the weighted blend of those presets:
  - freq       — pitch of the fundamental
  - brightness — harmonic richness (0 = pure sine, 1 = buzzy/full)
  - density    — pulse rate + sharpness of the amplitude envelope (the "beat")

Continuity: carrier and LFO phases accumulate across chunks, so there are no clicks when
the style — and therefore the params — drift.
"""

from __future__ import annotations

import numpy as np

from entrain.types import AudioChunk, StyleVector


class SynthEngine:
    ANCHOR_ORDER = ["calm", "groove", "peak"]  # must match configs/anchors.yaml order

    # keyword (substring of an anchor prompt) -> anchor name
    _KEYWORDS = {
        "calm": "calm", "ambient": "calm", "downtempo": "calm",
        "groov": "groove", "house": "groove", "deep": "groove",
        "peak": "peak", "techno": "peak", "driving": "peak",
    }
    # Wider spread than a "deep constant tone" — calm is a low soft drone, peak is a
    # bright fast-pulsing lead an octave-plus up, so the shift is obvious by ear.
    _PRESETS = {
        "calm":   dict(freq=98.00,  brightness=0.08, density=0.10),
        "groove": dict(freq=174.61, brightness=0.55, density=0.55),
        "peak":   dict(freq=329.63, brightness=0.95, density=0.95),
    }
    _N_HARMONICS = 8
    _GAIN = 0.2

    def __init__(self, sample_rate: int = 48_000, block: int = 2048) -> None:
        self.sample_rate = sample_rate
        self.block = block
        self._style: StyleVector | None = None
        self._tempo = 0.0          # BPM from movement; 0 -> fall back to density rate
        self._carrier_phase = 0.0
        self._lfo_phase = 0.0

    # --- MusicEngine interface --------------------------------------------------

    def embed_text(self, prompt: str) -> np.ndarray:
        """Anchor prompt -> one-hot over ANCHOR_ORDER (the synth's style basis)."""
        p = prompt.lower()
        vec = np.zeros(len(self.ANCHOR_ORDER), dtype=np.float32)
        for kw, anchor in self._KEYWORDS.items():
            if kw in p:
                vec[self.ANCHOR_ORDER.index(anchor)] = 1.0
                break
        return vec

    def start(self) -> None:
        self._carrier_phase = 0.0
        self._lfo_phase = 0.0

    def set_style(self, style: StyleVector) -> None:
        self._style = style

    def set_tempo(self, bpm: float) -> None:
        """Separate control channel (not in the style vector). 0 -> use the density rate."""
        self._tempo = float(bpm)

    def set_onset(self, onset: bool) -> None:
        pass                            # the synth has no drum channel

    def set_melody(self, pitch: int | None) -> None:
        pass                            # the synth has no note channel

    def next_chunk(self) -> AudioChunk:
        freq, brightness, density = self._params()
        sr, n = self.sample_rate, self.block
        t = np.arange(n, dtype=np.float64)

        # Carrier: harmonic stack, higher harmonics scaled by brightness.
        dphi = 2 * np.pi * freq / sr
        phase = self._carrier_phase + dphi * t
        sig = np.zeros(n)
        wsum = 0.0
        for h in range(1, self._N_HARMONICS + 1):
            amp = brightness ** (h - 1)
            sig += amp * np.sin(h * phase)
            wsum += amp
        sig /= max(wsum, 1e-6)
        self._carrier_phase = float((self._carrier_phase + dphi * n) % (2 * np.pi))

        # Amplitude envelope (the pulse). Rate locks to the movement tempo when we have
        # one (pulse-per-beat); otherwise it falls back to the density-derived rate.
        # Sharpness still rises with density, so the same tempo reads gentler when calm.
        lfo_freq = (self._tempo / 60.0) if self._tempo > 1.0 else (1.0 + 6.0 * density)
        dlfo = 2 * np.pi * lfo_freq / sr
        lphase = self._lfo_phase + dlfo * t
        pulse = (0.5 * (1 + np.sin(lphase))) ** (1.0 + 3.0 * density)
        env = (1 - 0.8 * density) + 0.8 * density * pulse
        self._lfo_phase = float((self._lfo_phase + dlfo * n) % (2 * np.pi))

        mono = (self._GAIN * sig * env).astype(np.float32)
        pcm = np.stack([mono, mono], axis=1)
        return AudioChunk(pcm=pcm, sample_rate=sr)

    def stop(self) -> None:
        self._style = None

    # --- internals --------------------------------------------------------------

    def _params(self) -> tuple[float, float, float]:
        """Blend the per-anchor presets by the current style weights."""
        vec = None if self._style is None else self._style.vec
        return self.params_from_weights(vec)

    @classmethod
    def params_from_weights(cls, weights) -> tuple[float, float, float]:
        """Pure: anchor weights -> (freq, brightness, density). Shared with the HUD so the
        screen shows exactly what the audio loop is rendering."""
        if weights is None:
            w = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # default: calm
        else:
            w = np.asarray(weights, dtype=np.float32)
        s = w.sum()
        w = w / s if s > 0 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        freq = bright = dens = 0.0
        for i, a in enumerate(cls.ANCHOR_ORDER):
            freq += w[i] * cls._PRESETS[a]["freq"]
            bright += w[i] * cls._PRESETS[a]["brightness"]
            dens += w[i] * cls._PRESETS[a]["density"]
        return float(freq), float(bright), float(dens)
