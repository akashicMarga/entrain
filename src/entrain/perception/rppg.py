"""rPPG -> heart rate. [FROZEN — classical DSP, no training]

Contactless cardiac signal from subtle facial colour change, using the POS algorithm
(Wang et al. 2017, "Plane-Orthogonal-to-Skin"). A face detector (BlazeFace) gives a
forehead ROI each frame; we buffer its mean RGB, project to the pulse plane, band-pass,
and take the spectral peak in the human-HR band (42-240 BPM).

Pure DSP, nothing trained. Close-range, single-face — degrades hard under motion, low
light, and distance, so the returned `confidence` gates how much to trust it. Heart rate
is feasible at 30 fps; HRV is marginal.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from entrain.perception import assets
from entrain.types import CardiacFeatures, Frame


def pos_heart_rate(rgb_means: np.ndarray, fps: float) -> tuple[float, float]:
    """POS over a window of ROI mean-RGB -> (bpm, confidence). NaN bpm if too weak."""
    n = len(rgb_means)
    if n < 8:
        return float("nan"), 0.0
    c = rgb_means.T.astype(np.float64)                     # (3, N)
    mean = c.mean(axis=1, keepdims=True)
    cn = c / (mean + 1e-9)                                 # temporal normalisation
    s = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]]) @ cn  # (2, N) pulse plane
    h = s[0] + (s[0].std() / (s[1].std() + 1e-9)) * s[1]
    h = h - h.mean()

    mag = np.abs(np.fft.rfft(h * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    band = (freqs >= 0.7) & (freqs <= 4.0)                 # 42-240 BPM
    if not band.any() or mag[band].sum() < 1e-9:
        return float("nan"), 0.0
    bm = mag[band]
    k = int(np.argmax(bm))
    bpm = float(freqs[band][k] * 60.0)
    confidence = float(bm[k] / (bm.sum() + 1e-9))          # spectral peak prominence
    return bpm, confidence


class RppgReader:
    """BlazeFace forehead ROI + POS heart-rate estimate over a sliding window."""

    def __init__(self, fps: int = 30, window_s: float = 8.0, min_confidence: float = 0.12) -> None:
        self.fps = fps
        self.window = max(16, int(fps * window_s))
        self.min_confidence = min_confidence
        self._fd = None
        self._roi_means: deque[np.ndarray] = deque(maxlen=self.window)
        self._hr_hist: deque[float] = deque(maxlen=max(2, int(fps * 2)))
        self._ts_ms = 0
        self._dt_ms = int(round(1000.0 / fps))

    def load(self) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        self._fd = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=assets.ensure("face_detector")),
                running_mode=vision.RunningMode.VIDEO,
            )
        )

    def __call__(self, frame: Frame) -> CardiacFeatures:
        import mediapipe as mp

        if self._fd is None:
            self.load()
        self._ts_ms += self._dt_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(frame.rgb))
        det = self._fd.detect_for_video(image, self._ts_ms)
        if not det.detections:
            self._roi_means.clear()                        # lost face -> restart buffer
            return CardiacFeatures(heart_rate=float("nan"), hr_trend=0.0, confidence=0.0)

        self._roi_means.append(self._forehead_rgb(frame.rgb, det.detections[0]))
        if len(self._roi_means) < self.window:
            return CardiacFeatures(heart_rate=float("nan"), hr_trend=0.0, confidence=0.0)

        bpm, conf = pos_heart_rate(np.asarray(self._roi_means), self.fps)
        if conf < self.min_confidence or not np.isfinite(bpm):
            return CardiacFeatures(heart_rate=float("nan"), hr_trend=0.0, confidence=conf)
        self._hr_hist.append(bpm)
        trend = float(self._hr_hist[-1] - self._hr_hist[0]) if len(self._hr_hist) > 1 else 0.0
        return CardiacFeatures(heart_rate=bpm, hr_trend=trend, confidence=conf)

    @staticmethod
    def _forehead_rgb(rgb: np.ndarray, detection) -> np.ndarray:
        """Mean RGB of a forehead patch inside the face bounding box."""
        bb = detection.bounding_box
        h, w = rgb.shape[:2]
        x0, y0 = bb.origin_x, bb.origin_y
        # forehead = upper-centre of the face box
        fx0 = int(np.clip(x0 + 0.30 * bb.width, 0, w - 1))
        fx1 = int(np.clip(x0 + 0.70 * bb.width, 1, w))
        fy0 = int(np.clip(y0 + 0.10 * bb.height, 0, h - 1))
        fy1 = int(np.clip(y0 + 0.25 * bb.height, 1, h))
        patch = rgb[fy0:fy1, fx0:fx1, :]
        if patch.size == 0:
            return np.zeros(3, dtype=np.float64)
        return patch.reshape(-1, 3).mean(axis=0).astype(np.float64)
