#!/usr/bin/env python
"""Launch the Stage 1 closed loop.

Tuning lives in configs/stage1.yaml (attack/release, sharpness, max_step, camera, engine,
source). CLI flags override the file; the file overrides the built-in defaults.

    python scripts/run_stage1.py                          # uses configs/stage1.yaml
    python scripts/run_stage1.py --source synthetic --headless   # no hardware, paced
    python scripts/run_stage1.py --config my.yaml --engine synth
"""

from __future__ import annotations

import argparse

from entrain.runtime.app import run_stage1
from entrain.runtime.config import DEFAULT_PATH, load_config


def main() -> None:
    p = argparse.ArgumentParser(description="Entrain — Stage 1 closed loop")
    p.add_argument("--config", default=DEFAULT_PATH, help="path to stage1.yaml")
    # CLI overrides (default None -> fall back to the config file).
    p.add_argument("--engine", choices=["synth", "selection", "mrt2"], default=None,
                   help="synth = audible now; selection = track library; mrt2 = generative")
    p.add_argument("--library-dir", default=None, help="track library (selection engine)")
    p.add_argument("--source", choices=["camera", "synthetic"], default=None,
                   help="camera = real webcam; synthetic = headless moving-blob source")
    p.add_argument("--headless", action="store_true",
                   help="use the silent paced sink instead of a real audio device")
    p.add_argument("--no-show", dest="show", action="store_false",
                   help="disable the live HUD window")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.engine is not None:
        cfg["engine"] = args.engine
    if args.library_dir is not None:
        cfg["library_dir"] = args.library_dir
    if args.source is not None:
        cfg["source"] = args.source

    run_stage1(config=cfg, headless=args.headless, show=args.show)


if __name__ == "__main__":
    main()
