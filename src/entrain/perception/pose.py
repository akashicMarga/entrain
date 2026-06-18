"""Movement sensing. [FROZEN — pretrained, no training]

Two readers behind one `PoseFeatures` contract:

  * `MediaPipePoseReader` — real skeletons via MediaPipe Tasks `PoseLandmarker` (multi-
    person). Yields motion energy (keypoint velocity), movement TEMPO (dominant rhythm of
    the body's vertical bob), and cross-subject SYNCHRONY (correlation between people's
    motion when 2+ are in frame — the design's true impact metric). This is the production
    Stage-1 reader; `PoseReader` aliases it.
  * `FrameDiffPoseReader` — the dependency-free frame-differencing fallback. No skeletons,
    so tempo/synchrony stay 0. Used with the synthetic camera and in headless tests.

The skeletons never leave this module — only the aggregated `PoseFeatures` do.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from entrain.perception import assets
from entrain.types import Frame, PoseFeatures

# Upper-body landmark indices (MediaPipe Pose, 33-point): nose, shoulders, wrists.
# Their vertical centroid is the "bob" signal we read tempo + synchrony from.
_UPPER_BODY = (0, 11, 12, 15, 16)


class _MotionTracker:
    """Per-pose history of the bob signal -> dominant tempo + pairwise synchrony."""

    def __init__(
        self,
        fps: int,
        history_s: float = 6.0,
        sync_window_s: float = 4.0,
        tempo_range_hz: tuple[float, float] = (0.5, 3.5),
        micro_window_s: float = 1.5,
        micro_gain: float = 250.0,
    ) -> None:
        self.fps = fps
        self.maxlen = max(8, int(fps * history_s))
        self.sync_n = max(8, int(fps * sync_window_s))
        self.lo, self.hi = tempo_range_hz
        # micromotion: a short window (responsive) with a high gain (the signal is tiny).
        self.micro_n = max(8, int(fps * micro_window_s))
        self.micro_gain = micro_gain
        self._sig: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=self.maxlen))

    def push(self, pose_idx: int, value: float) -> None:
        self._sig[pose_idx].append(value)

    def tempo_bpm(self, pose_idx: int = 0) -> float:
        sig = np.asarray(self._sig.get(pose_idx, ()), dtype=np.float64)
        if len(sig) < self.maxlen * 0.6:
            return 0.0
        x = sig - sig.mean()
        if not np.any(np.abs(x) > 1e-6):
            return 0.0
        mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), d=1.0 / self.fps)
        band = (freqs >= self.lo) & (freqs <= self.hi)
        if not band.any():
            return 0.0
        peak = freqs[band][int(np.argmax(mag[band]))]
        return float(peak * 60.0)

    def micromotion(self, pose_idx: int = 0) -> float:
        """Subtle-movement intensity: the windowed std of the DETRENDED bob signal.

        Gross ``motion_energy`` is a clipped per-frame velocity — it reads ~0 for a
        seated listener who is still but quietly engaged (small sway, micro-nods). This
        captures exactly that residual: subtract the window mean (drop the DC level so a
        static posture isn't movement), measure the fluctuation, scale up (the signal is
        tiny). The short window also bounds how much a slow lean leaks in. 0 when
        perfectly still or before the window fills. A dancer registers here too, but the
        estimator leans on it only when gross movement is low.
        """
        sig = np.asarray(self._sig.get(pose_idx, ()), dtype=np.float64)
        n = min(len(sig), self.micro_n)
        if n < self.micro_n:
            return 0.0
        w = sig[-n:]
        w = w - w.mean()                       # drop the DC level (static posture != movement)
        return float(np.clip(w.std() * self.micro_gain, 0.0, 1.0))

    def synchrony(self) -> float:
        """Positive Pearson correlation between pose 0 and pose 1 (0 if <2 people)."""
        a = np.asarray(self._sig.get(0, ()), dtype=np.float64)
        b = np.asarray(self._sig.get(1, ()), dtype=np.float64)
        n = min(len(a), len(b), self.sync_n)
        if n < self.sync_n:
            return 0.0
        a = a[-n:] - a[-n:].mean()
        b = b[-n:] - b[-n:].mean()
        da, db = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
        if da < 1e-6 or db < 1e-6:
            return 0.0
        return max(0.0, float((a * b).sum() / (da * db)))


class MediaPipePoseReader:
    """Real skeletons -> motion energy + tempo + synchrony."""

    def __init__(
        self,
        num_poses: int = 2,
        fps: int = 30,
        motion_gain: float = 12.0,
        history_s: float = 6.0,
        sync_window_s: float = 4.0,
        micro_gain: float = 250.0,
    ) -> None:
        self.num_poses = num_poses
        self.fps = fps
        self.motion_gain = motion_gain
        self._history_s = history_s
        self._sync_window_s = sync_window_s
        self._lm = None
        self._tracker = _MotionTracker(fps, history_s, sync_window_s, micro_gain=micro_gain)
        self._prev: dict[int, np.ndarray] = {}
        self._ts_ms = 0
        self._dt_ms = int(round(1000.0 / fps))

    def load(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=assets.ensure("pose_landmarker")),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=self.num_poses,
        )
        self._lm = vision.PoseLandmarker.create_from_options(opts)

    def __call__(self, frame: Frame) -> PoseFeatures:
        import mediapipe as mp

        if self._lm is None:
            self.load()
        self._ts_ms += self._dt_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(frame.rgb))
        result = self._lm.detect_for_video(image, self._ts_ms)
        poses = result.pose_landmarks  # list[ list[landmark] ]

        if not poses:
            self._prev.clear()
            return PoseFeatures(motion_energy=0.0, movement_tempo=0.0, synchrony=0.0)

        energies = []
        for i, lms in enumerate(poses):
            pts = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)  # (33, 2)
            if i in self._prev and self._prev[i].shape == pts.shape:
                energies.append(float(np.linalg.norm(pts - self._prev[i], axis=1).mean()))
            self._prev[i] = pts
            bob = float(pts[list(_UPPER_BODY), 1].mean())   # vertical centroid
            self._tracker.push(i, bob)

        energy = float(np.clip(np.mean(energies) * self.motion_gain, 0.0, 1.0)) if energies else 0.0
        # Lead hand = the higher wrist (15=left, 16=right); y is 0 at top, so raised = small y.
        wrist_y = np.array([poses[0][15].y, poses[0][16].y], dtype=np.float32)
        hand_height = float(np.clip(1.0 - wrist_y.min(), 0.0, 1.0))
        return PoseFeatures(
            motion_energy=energy,
            movement_tempo=self._tracker.tempo_bpm(0),
            synchrony=self._tracker.synchrony(),
            lead_hand_height=hand_height,
            micromotion=self._tracker.micromotion(0),
        )


class FrameDiffPoseReader:
    """Dependency-free frame-differencing motion energy (no skeletons)."""

    def __init__(self, sync_window_s: float = 10.0, motion_gain: float = 8.0,
                 fps: int = 30, micro_window_s: float = 1.5,
                 micro_gain: float = 80.0) -> None:
        self.sync_window_s = sync_window_s
        self.motion_gain = motion_gain
        self.micro_gain = micro_gain
        self._prev_gray: np.ndarray | None = None
        # short history of the changed-pixel fraction -> its windowed std is micromotion.
        self._changed: deque[float] = deque(maxlen=max(8, int(fps * micro_window_s)))

    def load(self) -> None:
        pass

    def __call__(self, frame: Frame) -> PoseFeatures:
        gray = frame.rgb.astype(np.float32).mean(axis=2) / 255.0
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return PoseFeatures(motion_energy=0.0, movement_tempo=0.0, synchrony=0.0)
        changed = float(np.abs(gray - self._prev_gray).mean())
        self._prev_gray = gray
        energy = float(np.clip(changed * self.motion_gain, 0.0, 1.0))
        self._changed.append(changed)
        micro = (
            float(np.clip(np.std(self._changed) * self.micro_gain, 0.0, 1.0))
            if len(self._changed) == self._changed.maxlen
            else 0.0
        )
        return PoseFeatures(motion_energy=energy, movement_tempo=0.0, synchrony=0.0,
                            micromotion=micro)


# Production Stage-1 reader.
PoseReader = MediaPipePoseReader
