"""Face -> affect (valence/arousal). [🪸 coral box #1 — heuristic now, trained later]

MediaPipe FaceLandmarker emits 52 facial blendshapes from PLAIN RGB (no TrueDepth / ARKit
needed — it derives its own from the image). Stage 1 maps those blendshapes to
valence/arousal with a small hand-written rule, so the affect channel is live today with
no training. The Stage-2 upgrade replaces `_blendshapes_to_va` with a trained MLX head on
the same blendshape vector (or the face crop) — the contract (`AffectFeatures`) is unchanged.

  valence  ~ smile - frown            (pleasant <-> unpleasant)
  arousal  ~ jaw-open, wide eyes, raised brows, smile energy   (calm <-> activated)

Crowd-averaged over detected faces. Degrades at distance — a desk/close-range signal.
"""

from __future__ import annotations

import numpy as np

from entrain.perception import assets
from entrain.types import AffectFeatures, Frame


class MediaPipeAffectReader:
    """FaceLandmarker blendshapes -> crowd-averaged valence/arousal."""

    def __init__(self, num_faces: int = 3, fps: int = 30) -> None:
        self.num_faces = num_faces
        self.fps = fps
        self._fl = None
        self._ts_ms = 0
        self._dt_ms = int(round(1000.0 / fps))

    def load(self) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        self._fl = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=assets.ensure("face_landmarker")),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=self.num_faces,
                output_face_blendshapes=True,
            )
        )

    def __call__(self, frame: Frame) -> AffectFeatures:
        import mediapipe as mp

        if self._fl is None:
            self.load()
        self._ts_ms += self._dt_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(frame.rgb))
        result = self._fl.detect_for_video(image, self._ts_ms)
        faces = result.face_blendshapes
        if not faces:
            return AffectFeatures(valence=0.0, arousal=0.0, n_faces=0)

        vas = [self._blendshapes_to_va({c.category_name: c.score for c in face})
               for face in faces]
        valence, arousal = np.mean(vas, axis=0)
        return AffectFeatures(valence=float(valence), arousal=float(arousal),
                              n_faces=len(faces))

    @staticmethod
    def _blendshapes_to_va(b: dict[str, float]) -> tuple[float, float]:
        """Hand-written blendshape -> (valence[-1,1], arousal[0,1]). Stage-2 trains this."""
        def g(*names: str) -> float:
            return float(np.mean([b.get(n, 0.0) for n in names]))

        smile = g("mouthSmileLeft", "mouthSmileRight")
        frown = g("mouthFrownLeft", "mouthFrownRight")
        sneer = g("noseSneerLeft", "noseSneerRight")
        valence = float(np.clip(1.6 * smile - 1.6 * frown - 0.8 * sneer, -1.0, 1.0))

        jaw = b.get("jawOpen", 0.0)
        eyes_wide = g("eyeWideLeft", "eyeWideRight")
        brows = g("browInnerUp", "browOuterUpLeft", "browOuterUpRight")
        arousal = float(np.clip(0.45 * jaw + 0.3 * eyes_wide + 0.25 * brows + 0.2 * smile,
                                0.0, 1.0))
        return valence, arousal


# Production Stage-1 reader.
AffectReader = MediaPipeAffectReader
