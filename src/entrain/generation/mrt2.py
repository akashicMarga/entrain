"""Magenta RealTime 2 adapter. [FROZEN — the live generative core]

MRT2 is an open-weights streaming model: it generates raw audio forward, frame by frame
(40 ms frames, 25 fps, 48 kHz stereo), conditioned on a MusicCoCa style embedding + its
own streaming state. "Real-time" here is literal — on Apple Silicon (MLX) it sustains a
continuous stream you steer mid-flight, which is exactly what the loop needs.

How our design plugs in cleanly:
  * `embed_text(prompt)` -> MRT2's MusicCoCa style embedding (a 768-d np.ndarray).
  * AnchorBank blends those embeddings by weight -> the style vector on the wire.
  * `generate(style=<blended vector>, frames=K, state=state)` tokenizes the blend
    internally and emits K frames, returning the updated state for the next call.

This is the library's own intended usage — MusicCoCa.tokenize is documented as operating
on `np.mean([promptA, promptB], axis=0)`, i.e. a blend of prompt embeddings.

Frozen: we never train MRT2. We embed anchors once, feed the blended style each chunk,
and keep the streaming state for continuity. Tempo travels via notes/MIDI, not the style
vector, so set_tempo is a no-op here for now.

Requires `pip install "magenta-rt[mlx]"` and `mrt models init && mrt models download`.
"""

from __future__ import annotations

import numpy as np

from entrain.types import AudioChunk, StyleVector


class DrumScheduler:
    """Combines a tempo-locked pulse with movement onset accents -> a per-chunk drum value.

    drums is one value per generate() call: -1 (masked, model's choice), 0 (no drum),
    1 (place a drum). Policy:
      * no rhythm and no accent  -> -1  (let the model do its own thing)
      * rhythmic / accenting     -> 1 on a beat boundary OR an onset, else 0
    So a steady bob lays down a pulse and a sharp accent drops an extra hit on top.
    """

    def __init__(self, chunk_dur: float, enabled: bool = True) -> None:
        self.chunk_dur = chunk_dur          # seconds of audio per generate() call
        self.enabled = enabled
        self._accum = 0.0                   # time since last beat
        self._pending_onset = False

    def trigger_onset(self) -> None:
        self._pending_onset = True

    def step(self, tempo: float) -> int:
        onset = self._pending_onset
        self._pending_onset = False
        if not self.enabled:
            return -1
        beat = False
        if tempo > 1.0:
            period = 60.0 / tempo
            self._accum += self.chunk_dur
            if self._accum >= period:
                beat = True
                self._accum -= period
        if tempo <= 1.0 and not onset:
            return -1                        # no groove yet -> hand drums back to the model
        return 1 if (beat or onset) else 0


class NoteScheduler:
    """Turns a single lead pitch into MRT2's 128-slot notes conditioning.

    A new pitch is an ONSET (2), a held pitch SUSTAINS (1), all other pitches are masked
    (-1, so the model is free to harmonise). No pitch -> all masked (the model's own melody).
    """

    _NUM_PITCHES = 128

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._prev_pitch: int | None = None

    def step(self, pitch: int | None) -> list[int] | None:
        if not self.enabled:
            return None
        notes = [-1] * self._NUM_PITCHES
        if pitch is not None and 0 <= pitch < self._NUM_PITCHES:
            notes[pitch] = 2 if pitch != self._prev_pitch else 1   # onset vs sustain
        self._prev_pitch = pitch
        return notes


class Mrt2Engine:
    """Streaming MRT2 behind the MusicEngine interface."""

    def __init__(
        self,
        size: str = "mrt2_base",        # "mrt2_base" (2.4B) or "mrt2_small" (230M)
        frames_per_chunk: int = 10,     # 10 frames = 400 ms per next_chunk()
        temperature: float = 1.3,
        top_k: int = 40,
        cfg_musiccoca: float = 3.0,
        bits: int | None = None,        # quantization for the non-mlxfn path
        use_mlxfn: bool = True,         # exported .mlxfn graph = faster
        drums_enabled: bool = False,    # movement-driven drum channel
        cfg_drums: float = 4.0,         # how hard MRT2 obeys the drum hint
        notes_enabled: bool = False,    # hand-driven melody (notes) channel
        cfg_notes: float = 2.0,         # how hard MRT2 obeys the note hint
    ) -> None:
        self.size = size
        self.frames_per_chunk = frames_per_chunk
        self.temperature = temperature
        self.top_k = top_k
        self.cfg_musiccoca = cfg_musiccoca
        self.cfg_drums = cfg_drums
        self.cfg_notes = cfg_notes
        self.bits = bits
        self.use_mlxfn = use_mlxfn
        self.sample_rate = 48_000
        self._mrt = None
        self._state = None
        self._style_vec: np.ndarray | None = None
        self._tempo = 0.0
        self._drums = DrumScheduler(
            chunk_dur=frames_per_chunk * 0.04, enabled=drums_enabled
        )
        self._notes = NoteScheduler(enabled=notes_enabled)
        self._pitch: int | None = None
        self.last_drum = -1             # most recent drum value (for the HUD)

    def _ensure_model(self) -> None:
        if self._mrt is not None:
            return
        if self.use_mlxfn:
            from magenta_rt import MagentaRT2Mlxfn

            self._mrt = MagentaRT2Mlxfn(
                size=self.size, temperature=self.temperature, top_k=self.top_k,
                cfg_musiccoca=self.cfg_musiccoca,
            )
        else:
            from magenta_rt import MagentaRT2Mlx

            self._mrt = MagentaRT2Mlx(
                size=self.size, temperature=self.temperature, top_k=self.top_k,
                cfg_musiccoca=self.cfg_musiccoca, bits=self.bits,
            )

    # --- MusicEngine interface --------------------------------------------------

    def embed_text(self, prompt: str) -> np.ndarray:
        """Anchor prompt -> MusicCoCa style embedding (768-d). Called once per anchor."""
        self._ensure_model()
        return np.asarray(self._mrt.embed_style(prompt, use_mapper=True), dtype=np.float32)

    def start(self) -> None:
        self._ensure_model()
        self._state = None              # fresh streaming state

    def set_style(self, style: StyleVector) -> None:
        self._style_vec = np.asarray(style.vec, dtype=np.float32)

    def set_tempo(self, bpm: float) -> None:
        self._tempo = float(bpm)        # drives the tempo-locked drum pulse

    def set_onset(self, onset: bool) -> None:
        if onset:
            self._drums.trigger_onset()  # a movement accent -> a drum hit

    def set_melody(self, pitch: int | None) -> None:
        self._pitch = pitch              # lead pitch from hand height

    def next_chunk(self) -> AudioChunk:
        self._ensure_model()
        drum = self._drums.step(self._tempo)   # -1 masked / 0 none / 1 hit
        self.last_drum = drum
        notes = self._notes.step(self._pitch)  # 128-slot notes, or None
        wav, self._state = self._mrt.generate(
            style=self._style_vec,      # None -> unconditional until first set_style
            drums=[drum],               # length-1 drum conditioning
            cfg_drums=self.cfg_drums,
            notes=notes,                # hand-played lead, or None (model's own melody)
            cfg_notes=self.cfg_notes,
            frames=self.frames_per_chunk,
            state=self._state,
        )
        pcm = np.asarray(wav.samples, dtype=np.float32)
        if pcm.ndim == 1:
            pcm = np.stack([pcm, pcm], axis=1)
        return AudioChunk(pcm=pcm, sample_rate=self.sample_rate)

    def stop(self) -> None:
        self._state = None
