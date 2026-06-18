"""Camera capture. [WORLD — OS/hardware, no ML]

Owns the capture session and yields `Frame`s at ~30 fps. Two sources, same interface:

  * `CameraSource`    — real webcam via OpenCV (`cv2.VideoCapture`). Needs `opencv-python`.
  * `SyntheticCamera` — a headless frame source (a moving bright blob) for running and
    testing the whole loop with no webcam and no extra deps. Faster blob -> more changed
    pixels -> higher motion energy, so it exercises perception + policy end to end.

IMPORTANT (real camera): disable Center Stage / auto-framing / auto-processing — it stomps
the tiny rPPG signal. The laptop camera is a single-subject, desk-distance sensor; a
wider-FOV, low-light camera is what a real crowd needs later. Algorithms port; the sensor
changes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import numpy as np

from entrain.types import Frame


class CameraSource:
    """Real webcam. Yields `Frame`s from an OpenCV `VideoCapture`."""

    def __init__(self, device: int = 0, fps: int = 30) -> None:
        self.device = device
        self.fps = fps
        self._cap = None
        self._frame_id = 0

    def open(self) -> None:
        import cv2

        self._cap = cv2.VideoCapture(self.device)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open camera device {self.device}")

    def frames(self) -> Iterator[Frame]:
        if self._cap is None:
            self.open()
        while True:
            ok, bgr = self._cap.read()
            if not ok:
                break
            rgb = np.ascontiguousarray(bgr[..., ::-1])  # BGR -> RGB
            yield Frame(rgb=rgb, t=time.monotonic(), frame_id=self._frame_id)
            self._frame_id += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class SyntheticCamera:
    """Headless frame source: a bright vertical bar that drifts across the frame.

    `motion` is pixels/frame — a float, or a zero-arg callable for a time-varying signal
    (so a test can script "still" then "dancing"). Useful for running the full loop with
    no hardware.
    """

    def __init__(
        self,
        size: tuple[int, int] = (120, 160),
        motion: float | Callable[[], float] = 3.0,
        bar_half_width: int = 5,
        fps: int = 30,
        n_frames: int | None = None,
    ) -> None:
        self.h, self.w = size
        self.motion = motion
        self.bar_half_width = bar_half_width
        self.fps = fps
        self.n_frames = n_frames
        self._frame_id = 0
        self._x = float(self.w // 2)

    def open(self) -> None:
        pass

    def frames(self) -> Iterator[Frame]:
        while self.n_frames is None or self._frame_id < self.n_frames:
            step = self.motion() if callable(self.motion) else self.motion
            self._x = (self._x + step) % self.w
            img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            x = int(self._x) % self.w
            lo, hi = max(0, x - self.bar_half_width), x + self.bar_half_width
            img[:, lo:hi, :] = 255
            yield Frame(rgb=img, t=self._frame_id / self.fps, frame_id=self._frame_id)
            self._frame_id += 1

    def close(self) -> None:
        pass
