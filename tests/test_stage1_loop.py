"""Stage 1 end-to-end-ish tests: the real DSP boxes + the closed vision loop.

No webcam, no audio device. The synthetic camera drives motion; we assert the perception
-> state -> policy -> anchors -> slot chain moves the style vector toward the high-energy
anchor when motion is high.
"""

import numpy as np

from entrain.capture.camera import SyntheticCamera
from entrain.generation.synth import SynthEngine
from entrain.perception.pose import FrameDiffPoseReader
from entrain.policy.anchors import AnchorBank
from entrain.policy.heuristic import HeuristicPolicy
from entrain.runtime.loops import VisionLoop
from entrain.runtime.shared_state import StyleSlot
from entrain.state.estimator import StateEstimator
from entrain.types import StyleVector


# --- synth engine ---------------------------------------------------------------

def test_embed_text_is_one_hot_by_keyword():
    eng = SynthEngine()
    np.testing.assert_array_equal(eng.embed_text("calm ambient downtempo"), [1, 0, 0])
    np.testing.assert_array_equal(eng.embed_text("groovy deep house"), [0, 1, 0])
    np.testing.assert_array_equal(eng.embed_text("peak-time techno, driving"), [0, 0, 1])


def test_synth_chunk_shape_and_range():
    eng = SynthEngine(block=1024)
    eng.start()
    eng.set_style(StyleVector(vec=np.array([0, 0, 1], dtype=np.float32)))
    chunk = eng.next_chunk()
    assert chunk.pcm.shape == (1024, 2)
    assert chunk.pcm.dtype == np.float32
    assert np.isfinite(chunk.pcm).all()
    assert np.abs(chunk.pcm).max() <= SynthEngine._GAIN + 1e-6


def test_synth_is_click_free_across_chunks():
    """Phase accumulates, so the seam between consecutive chunks is continuous."""
    eng = SynthEngine(block=1024)
    eng.start()
    eng.set_style(StyleVector(vec=np.array([0, 1, 0], dtype=np.float32)))
    c1 = eng.next_chunk().pcm[:, 0]
    c2 = eng.next_chunk().pcm[:, 0]
    seam = abs(float(c2[0] - c1[-1]))
    within = float(np.abs(np.diff(c1)).max())
    assert seam <= within * 2 + 1e-4   # no jump bigger than a normal sample-to-sample step


def test_peak_style_is_brighter_than_calm():
    eng = SynthEngine(block=4096)
    eng.start()
    eng.set_style(StyleVector(vec=np.array([1, 0, 0], dtype=np.float32)))
    calm = eng.next_chunk().pcm[:, 0]
    eng.set_style(StyleVector(vec=np.array([0, 0, 1], dtype=np.float32)))
    peak = eng.next_chunk().pcm[:, 0]
    # peak = higher fundamental + more harmonics -> higher spectral centroid.
    assert _spectral_centroid(peak) > _spectral_centroid(calm)


def _spectral_centroid(x: np.ndarray) -> float:
    win = np.hanning(len(x))                       # suppress leakage
    power = np.abs(np.fft.rfft(x * win)) ** 2      # power, not magnitude
    freqs = np.fft.rfftfreq(len(x))
    return float((freqs * power).sum() / (power.sum() + 1e-9))


# --- pose motion energy ---------------------------------------------------------

def test_pose_energy_rises_with_motion():
    def mean_energy(motion: float) -> float:
        reader = FrameDiffPoseReader()
        cam = SyntheticCamera(motion=motion, n_frames=10)
        es = [reader(f).motion_energy for f in cam.frames()]
        return float(np.mean(es[1:]))  # skip first (primes the prev-frame buffer)

    assert mean_energy(20.0) > mean_energy(1.0)


# --- the closed vision loop -----------------------------------------------------

def test_vision_loop_steers_toward_peak_with_motion():
    def run_with_motion(motion: float) -> dict:
        slot = StyleSlot(max_step=1.0)            # let it converge fast for the test
        engine = SynthEngine()
        anchors = AnchorBank(embed_fn=engine.embed_text)
        # build the AnchorSet without touching the yaml file:
        from entrain.types import AnchorSet
        names = SynthEngine.ANCHOR_ORDER
        anchors._anchors = AnchorSet(
            names=names, embeddings=np.eye(len(names), dtype=np.float32)
        )
        policy = HeuristicPolicy(anchor_names=names)
        cam = SyntheticCamera(motion=motion, n_frames=60)
        loop = VisionLoop(cam, FrameDiffPoseReader(), StateEstimator(attack=0.5, release=0.1), policy, anchors, slot)
        loop.run()
        return slot.read().weights

    still = run_with_motion(0.0)
    dancing = run_with_motion(18.0)
    assert dancing["peak"] > still["peak"]
    assert dancing["peak"] > dancing["calm"]


# --- director snapshots (before/after capture for a future VLM) -----------------

def test_vision_loop_captures_before_after_snapshots_on_directive_change():
    from entrain.runtime.episode import EpisodeBuffer
    from entrain.types import AnchorSet, AnchorSpec, Directive, Frame, PoseFeatures

    class _Cam:                       # yields timed frames, then ends the loop
        def frames(self):
            for i in range(20):
                yield Frame(rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                            t=i * 0.1, frame_id=i)         # dt=0.1s

    class _Pose:                      # constant movement; loop never calls .load()
        def __call__(self, frame):
            return PoseFeatures(motion_energy=0.5, movement_tempo=0.0, synchrony=0.3)

    class _Director:                  # forces exactly one directive change
        def __init__(self):
            self._done = False
        def initial(self):
            return Directive(anchors=[AnchorSpec("calm", "c")], intent="neutral groove")
        def revise(self, ctx):
            if self._done:
                return None
            self._done = True
            return Directive(anchors=[AnchorSpec("calm", "c")], intent="building")

    names = SynthEngine.ANCHOR_ORDER
    anchors = AnchorBank(embed_fn=SynthEngine().embed_text)
    anchors._anchors = AnchorSet(names=names, embeddings=np.eye(len(names), dtype=np.float32))
    ep = EpisodeBuffer(fps=10)
    loop = VisionLoop(
        _Cam(), _Pose(), StateEstimator(), HeuristicPolicy(anchor_names=names),
        anchors, StyleSlot(max_step=1.0),
        director=_Director(), episode=ep, revise_every_s=0.3,   # tick fast for the test
    )
    loop.run()

    labels = [s.label for s in ep.snapshots]
    assert any(l == "before:building" for l in labels)   # captured the room that prompted it
    assert any(l == "after:building" for l in labels)     # ... and the room after the lag
    assert all(s.frame is not None for s in ep.snapshots) # frames stored for the VLM
    assert all(s.frame.shape == (2, 2, 3) for s in ep.snapshots)  # thumbnailed (8/4)
