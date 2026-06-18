"""Integration smoke test for the real MediaPipe reader.

Skipped automatically when mediapipe or the model asset isn't available (e.g. CI), so the
rest of the suite stays dependency-free. On a dev machine it confirms the reader builds,
runs on a frame, and returns a valid PoseFeatures (a blank frame has no person -> zeros).
"""

import numpy as np
import pytest

mp = pytest.importorskip("mediapipe")

from entrain.perception import assets
from entrain.perception.pose import MediaPipePoseReader
from entrain.types import Frame

pytestmark = pytest.mark.skipif(
    not assets.is_available("pose_landmarker"),
    reason="pose landmarker model asset not downloaded",
)


def test_reader_runs_on_blank_frame():
    reader = MediaPipePoseReader(num_poses=2, fps=30)
    reader.load()
    blank = Frame(rgb=np.zeros((240, 320, 3), dtype=np.uint8), t=0.0, frame_id=0)
    feats = reader(blank)
    assert feats.motion_energy == 0.0          # no person detected
    assert feats.movement_tempo == 0.0
    assert feats.synchrony == 0.0
    # a second frame must not raise (timestamps strictly increase internally)
    reader(Frame(rgb=np.zeros((240, 320, 3), dtype=np.uint8), t=1 / 30, frame_id=1))
