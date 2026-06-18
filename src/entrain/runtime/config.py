"""Stage 1 config loader. [BUILD]

Reads configs/stage1.yaml and merges it over built-in defaults, so a missing file or a
missing key is never fatal — the defaults here are the source of truth, the file only
overrides. CLI flags in turn override the loaded config (resolved in scripts/run_stage1.py).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "engine": "synth",
    "library_dir": None,
    "source": "camera",
    # responsive=False freezes steering after warmup -> static-generative A/B baseline.
    "responsive": True,
    # MRT2 streaming generator (engine: mrt2). 2.4B "mrt2_base" needs a Pro/Max chip for
    # real-time; "mrt2_small" (230M) runs on any Apple Silicon.
    "mrt2": {"size": "mrt2_small", "frames_per_chunk": 4, "temperature": 1.3,
             "top_k": 40, "cfg_musiccoca": 3.0, "audio_buffer": 2,
             # movement-driven drums: tempo-locked pulse + onset accents. Thresholds are
             # CALIBRATED per-session (SignalCalibrator) -> "above your own resting level",
             # not absolute, so they don't fire constantly on a still seated bob.
             "drums": True, "cfg_drums": 4.0,
             "drum_gate": 0.1,          # min movement above rest (motion-energy units) to pulse
             "onset_rise_sigma": 1.5,   # accent = a jump this many spreads above your usual
             "onset_floor": 1.0,        # ... and only while at least this far above rest
             "onset_refractory": 5,
             # hand-driven melody: raise your hand to play a pentatonic lead
             "melody": True, "cfg_notes": 2.0,
             "melody_root": 48, "melody_octaves": 2, "melody_threshold": 0.35},
    "camera": {"device": 0, "fps": 30},
    # pose: real skeletons (mediapipe) or dependency-free frame-diff fallback.
    # synthetic source forces frame-diff regardless (a blob isn't a person).
    "pose": {"backend": "mediapipe", "num_poses": 2, "motion_gain": 12.0,
             "micro_gain": 250.0},
    # face affect (valence/arousal from blendshapes) and contactless heart rate (POS).
    # both are close-range face signals -> auto-skipped for the synthetic source.
    "affect": {"enabled": True, "num_faces": 3},
    "rppg": {"enabled": True, "window_s": 8.0},
    # asymmetric smoothing — fast rise, slow fall ("hold the vibe")
    "state": {"attack": 0.45, "release": 0.04, "micro_weight": 0.6},
    "policy": {"sharpness": 4.0, "synchrony_weight": 0.0,
               "valence_weight": 0.25, "hr_weight": 0.2},
    "style_slot": {"max_step": 0.12},
}

DEFAULT_PATH = "configs/stage1.yaml"


def load_config(path: str | None = DEFAULT_PATH) -> dict[str, Any]:
    """Defaults, with the yaml file (if present) merged on top."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        p = Path(path)
        if p.exists():
            import yaml

            loaded = yaml.safe_load(p.read_text()) or {}
            _deep_update(cfg, loaded)
    return cfg


def _deep_update(base: dict, over: dict) -> None:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
