"""The two decoupled loops. [BUILD — the make-or-break runtime discipline]

A fast AUDIO loop that must never stall, and a slow VISION loop that updates the shared
style slot asynchronously. The audio loop always reads the freshest vector; the camera
never blocks the sound. Worst case is stale steering, never silence.

    audio loop (fast):   read slot -> engine.next_chunk() -> audio sink        [never stalls]
    vision loop (slow):  capture -> perception -> state -> policy -> write slot [may lag]
"""

from __future__ import annotations

import threading

from entrain.capture.camera import CameraSource
from entrain.generation.engine import MusicEngine
from entrain.output.audio_sink import AudioSink
from entrain.perception.affect import AffectReader
from entrain.perception.pose import PoseReader
from entrain.perception.rppg import RppgReader
from entrain.policy.anchors import AnchorBank
from entrain.policy.base import Policy
from entrain.runtime.shared_state import StyleSlot
from entrain.state.estimator import StateEstimator


class AudioLoop:
    """Fast loop. Pulls chunks from the engine using the freshest style vector."""

    def __init__(self, engine: MusicEngine, sink: AudioSink, slot: StyleSlot,
                 warmup=None) -> None:
        self.engine = engine
        self.sink = sink
        self.slot = slot
        # warmup() runs ON THIS THREAD before the loop — model load + anchor embedding.
        # Keeping it here means all MLX work (load, embed, generate) shares one thread,
        # which MLX requires (streams are thread-local).
        self.warmup = warmup
        self._stop = threading.Event()

    def run(self) -> None:
        self.engine.start()
        if self.warmup is not None:
            self.warmup()
        self.sink.open()
        while not self._stop.is_set():
            style = self.slot.read()
            if style is not None:
                self.engine.set_style(style)
            self.engine.set_tempo(self.slot.tempo)      # tempo-locked drum pulse
            self.engine.set_onset(self.slot.take_onset())  # movement accents -> drum hits
            self.engine.set_melody(self.slot.melody)    # hand height -> lead pitch
            chunk = self.engine.next_chunk()            # blocking only on the sink's pace
            if hasattr(self.engine, "last_drum"):
                self.slot.set_drum(self.engine.last_drum)  # surface hits to the HUD
            self.sink.write(chunk)

    def stop(self) -> None:
        self._stop.set()


class VisionLoop:
    """Slow loop. Senses the crowd and writes a new style vector to the slot."""

    def __init__(
        self,
        camera: CameraSource,
        pose: PoseReader,
        state: StateEstimator,
        policy: Policy,
        anchors: AnchorBank,
        slot: StyleSlot,
        affect: AffectReader | None = None,
        rppg: RppgReader | None = None,
        on_step=None,
        ready: threading.Event | None = None,
        onset_detector=None,
        melody_mapper=None,
        calibrator=None,
        drum_gate: float = 0.1,
    ) -> None:
        self.camera = camera
        self.pose = pose
        self.affect = affect
        self.rppg = rppg
        self.state = state
        self.policy = policy
        self.anchors = anchors
        self.slot = slot
        # Optional movement-accent detector (drives drum hits). Fed a CALIBRATED level
        # (movement above this person's resting floor, in spread units) when a calibrator
        # is present, else raw motion energy.
        self.onset_detector = onset_detector
        # Per-session movement calibration (auto-zeroes idle jitter / camera baseline).
        self.calibrator = calibrator
        # Drum pulse gate: lay the tempo-locked pulse only when movement above rest clears
        # this (native motion-energy units); below it, rhythm is handed back to the model.
        self.drum_gate = drum_gate
        # Optional hand-height -> pitch mapper (drives the melody/notes channel).
        self.melody_mapper = melody_mapper
        # Optional observer: on_step(frame, state, style) -> bool. Return False to stop.
        # Used by the HUD; keeps cv2 out of the loop itself.
        self.on_step = on_step
        # Set by the audio loop once anchors are embedded (they're embedded on the audio
        # thread for MLX thread-affinity); blending can't happen until then.
        self.ready = ready
        self._stop = threading.Event()
        self._warned = False

    def run(self) -> None:
        if self.ready is not None:
            self.ready.wait()
        for frame in self.camera.frames():
            if self._stop.is_set():
                break
            # A single bad frame (e.g. a MediaPipe hiccup) must NOT kill perception — that
            # would silently freeze the music. Log once and keep going.
            try:
                pose_f = self.pose(frame)
                affect_f = self.affect(frame) if self.affect else None
                cardiac_f = self.rppg(frame) if self.rppg else None
                # Calibrate raw motion energy to this person's resting level, so accents
                # and the drum pulse fire on real movement, not idle jitter / phantom motion.
                excess = pose_f.motion_energy
                onset_level = pose_f.motion_energy
                if self.calibrator is not None:
                    self.calibrator.update(pose_f.motion_energy)
                    excess = self.calibrator.excess(pose_f.motion_energy)
                    onset_level = excess / self.calibrator.spread   # jump in "your usual" units
                # Onset on the (calibrated) RAW level, before smoothing -> a drum accent.
                if self.onset_detector is not None and self.onset_detector(onset_level):
                    self.slot.set_onset()
                if self.melody_mapper is not None:
                    self.slot.set_melody(self.melody_mapper(pose_f.lead_hand_height))
                state = self.state.update(pose_f, affect_f, cardiac_f)
                weights = self.policy(state)
                style = self.anchors.blend(weights)
                # Gate the tempo-locked drum pulse on real movement above rest; otherwise
                # send tempo 0 so the DrumScheduler hands rhythm back to the model (no
                # phantom pulse from a spurious bob tempo while you sit still).
                tempo = state.tempo
                if self.calibrator is not None and excess < self.drum_gate:
                    tempo = 0.0
                self.slot.write(style, tempo=tempo)
            except Exception as e:
                if not self._warned:
                    self._warned = True
                    print(f"[entrain] perception error (continuing): {e!r}")
                continue
            if self.on_step is not None and self.on_step(frame, state, self.slot.read()) is False:
                break

    def stop(self) -> None:
        self._stop.set()
