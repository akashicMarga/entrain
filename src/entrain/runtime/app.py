"""Stage 1 wiring. [BUILD — constructs and runs both loops]

Assembles the whole perceive -> decide -> generate loop and runs the two threads. The forks
are all one line here:
  * engine:  synth (audible today) | selection (de-risked) | mrt2 (generative)
  * policy:  HeuristicPolicy (Stage 1) | LearnedPolicy (Stage 3)
  * source:  real webcam | synthetic (headless)
  * sink:    real audio out | null (headless, paced)
"""

from __future__ import annotations

import threading
from typing import Any

from entrain.capture.camera import CameraSource, SyntheticCamera
from entrain.generation.mrt2 import Mrt2Engine
from entrain.generation.selection import SelectionEngine
from entrain.generation.synth import SynthEngine
from entrain.output.audio_sink import AudioSink, NullSink, QueuedAudioSink
from entrain.perception.affect import AffectReader
from entrain.perception.pose import FrameDiffPoseReader, MediaPipePoseReader
from entrain.perception.rppg import RppgReader
from entrain.policy.anchors import AnchorBank
from entrain.policy.director import StaticDirector
from entrain.policy.heuristic import HeuristicPolicy
from entrain.runtime.config import load_config
from entrain.runtime.loops import AudioLoop, VisionLoop
from entrain.runtime.shared_state import StyleSlot
from entrain.state.estimator import StateEstimator


def build_engine(cfg: dict[str, Any]):
    kind = cfg["engine"]
    if kind == "synth":
        return SynthEngine()
    if kind == "selection":
        if not cfg["library_dir"]:
            raise ValueError("selection engine needs library_dir")
        return SelectionEngine(cfg["library_dir"], embed_audio_fn=None)
    if kind == "mrt2":
        m = cfg["mrt2"]
        return Mrt2Engine(
            size=m["size"], frames_per_chunk=m["frames_per_chunk"],
            temperature=m["temperature"], top_k=m["top_k"],
            cfg_musiccoca=m["cfg_musiccoca"],
            drums_enabled=m["drums"], cfg_drums=m["cfg_drums"],
            notes_enabled=m["melody"], cfg_notes=m["cfg_notes"],
        )
    raise ValueError(f"unknown engine: {kind}")


def run_stage1(
    config: dict[str, Any] | None = None,
    headless: bool = False,
    show: bool = True,
) -> None:
    cfg = config if config is not None else load_config()
    cam_cfg = cfg["camera"]
    st_cfg = cfg["state"]
    pol_cfg = cfg["policy"]

    slot = StyleSlot(max_step=cfg["style_slot"]["max_step"], responsive=cfg["responsive"])

    # Output stage (frozen renderer). Anchor NAMES are read now (for the policy); the
    # actual embedding is deferred to the audio thread (warmup) so all MLX work — model
    # load, anchor embed, generation — shares one thread (MLX streams are thread-local).
    engine = build_engine(cfg)
    # The slow DIRECTOR layer authors the anchor set; the fast policy weights over it.
    # Stage 1 uses StaticDirector (the fixed configs/anchors.yaml anchors) — swap in
    # LlmDirector to let a small LLM author prompts + arc on a seconds-scale cadence.
    director = StaticDirector()
    anchors = AnchorBank(embed_fn=engine.embed_text, directive=director.initial())
    policy = HeuristicPolicy(                                 # Stage 1: no training
        anchor_names=anchors.names,
        sharpness=pol_cfg["sharpness"],
        synchrony_weight=pol_cfg["synchrony_weight"],
        valence_weight=pol_cfg["valence_weight"],
        hr_weight=pol_cfg["hr_weight"],
    )

    # Perception. Movement always; face affect + heart rate when a real camera sees a face.
    synthetic = cfg["source"] == "synthetic"
    camera = (
        SyntheticCamera()
        if synthetic
        else CameraSource(device=cam_cfg["device"], fps=cam_cfg["fps"])
    )
    camera.open()

    # Synthetic blobs aren't people -> frame-diff. Real camera -> config's pose backend.
    pose_cfg = cfg["pose"]
    if synthetic or pose_cfg["backend"] == "framediff":
        pose = FrameDiffPoseReader(motion_gain=pose_cfg["motion_gain"], fps=cam_cfg["fps"])
    else:
        pose = MediaPipePoseReader(
            num_poses=pose_cfg["num_poses"],
            fps=cam_cfg["fps"],
            motion_gain=pose_cfg["motion_gain"],
            micro_gain=pose_cfg["micro_gain"],
        )
    pose.load()

    # Face channels — only meaningful on a real camera (a blob has no face).
    affect = rppg = None
    if not synthetic:
        if cfg["affect"]["enabled"]:
            affect = AffectReader(num_faces=cfg["affect"]["num_faces"], fps=cam_cfg["fps"])
            affect.load()
        if cfg["rppg"]["enabled"]:
            rppg = RppgReader(fps=cam_cfg["fps"], window_s=cfg["rppg"]["window_s"])
            rppg.load()

    state = StateEstimator(attack=st_cfg["attack"], release=st_cfg["release"],
                           micro_weight=st_cfg["micro_weight"])

    # Streaming generation needs the generate-ahead queued sink to stay gap-free;
    # the cheap local synth is fine on the simple blocking sink.
    if headless:
        sink = NullSink()
    elif cfg["engine"] == "mrt2":
        sink = QueuedAudioSink(max_queue=cfg["mrt2"]["audio_buffer"])
    else:
        sink = AudioSink()
    # Embed anchors on the audio thread (warmup), then signal the vision loop it can blend.
    anchors_ready = threading.Event()

    def warmup() -> None:
        anchors.embed()
        anchors_ready.set()

    audio = AudioLoop(engine, sink, slot, warmup=warmup)

    # HUD (optional) runs on the MAIN thread; the audio loop is the background thread, so
    # the window never blocks the sound. Without a HUD, both loops are background threads.
    hud = None
    if show and not headless:
        from entrain.viz.hud import Hud

        hud = Hud(slot=slot)
    # Movement -> drums (accents) and hand -> melody (notes), only for MRT2.
    onset_detector = melody_mapper = calibrator = None
    drum_gate = 0.1
    if cfg["engine"] == "mrt2":
        m = cfg["mrt2"]
        if m["drums"]:
            from entrain.perception.onset import OnsetDetector
            from entrain.state.calibration import SignalCalibrator

            # Calibrate movement per-session so the accent + pulse thresholds are
            # "above YOUR resting level" rather than fixed constants (the fix for drums
            # firing constantly on a spurious seated-bob tempo).
            calibrator = SignalCalibrator()
            drum_gate = m["drum_gate"]
            onset_detector = OnsetDetector(
                threshold=m["onset_rise_sigma"], refractory=m["onset_refractory"],
                floor=m["onset_floor"],
            )
        if m["melody"]:
            from entrain.perception.melody import MelodyMapper

            melody_mapper = MelodyMapper(
                root=m["melody_root"], octaves=m["melody_octaves"],
                active_threshold=m["melody_threshold"],
            )

    vision = VisionLoop(
        camera, pose, state, policy, anchors, slot,
        affect=affect, rppg=rppg,
        on_step=(hud.render if hud else None),
        ready=anchors_ready,
        onset_detector=onset_detector,
        melody_mapper=melody_mapper,
        calibrator=calibrator,
        drum_gate=drum_gate,
    )

    t_audio = threading.Thread(target=audio.run, name="audio", daemon=True)
    t_audio.start()
    try:
        if hud is not None:
            vision.run()                 # main thread (cv2 GUI)
        else:
            t_vision = threading.Thread(target=vision.run, name="vision", daemon=True)
            t_vision.start()
            t_vision.join()
    except KeyboardInterrupt:
        pass
    finally:
        vision.stop()
        audio.stop()
        camera.close()
        if hud is not None:
            hud.close()
