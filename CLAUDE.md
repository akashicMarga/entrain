# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Entrain is a real-time closed-loop music system: a camera senses how people respond to music and steers what plays. The loop is:

```
camera → perception → state estimate → control policy → generation → music → crowd → (back to camera)
```

The two parts with learnable weights ("coral boxes") are **facial affect read-out** (`perception/affect.py`) and the **control policy** (`policy/learned.py`) — both MLX, both mostly future work. Everything else is pretrained (pose), classical DSP (rPPG), or a frozen renderer (the generator). You do **not** train a music generator.

This is a staged build: **Stage 1** = zero training, hand-written heuristic policy, prove the loop end-to-end on a MacBook (this is what runs today). Stage 2 = supervised affect adaptation. Stage 3 = the learned policy. See `README.md` (build plan) and `docs/architecture.md` (module contracts).

## Commands

```bash
pip install -e .                  # core deps
pip install -e ".[dev]"           # + pytest
pip install -e ".[mrt2]"          # + magenta-rt[mlx] for the generative engine

pytest                            # full suite (pythonpath=src is set in pyproject.toml)
pytest tests/test_pipeline_seams.py::test_anchor_blend_is_normalized_weighted_sum   # single test

python scripts/run_stage1.py                                   # synth engine + live HUD (needs camera + audio)
python scripts/run_stage1.py --source synthetic --headless     # no hardware, paced — the CI-safe smoke run
python scripts/run_stage1.py --engine mrt2                     # generative (after: mrt models init && mrt models download mrt2_small)
entrain-stage1                                                  # console-script entry point (entrain.runtime.app:run_stage1)
```

CLI flags (`--engine`, `--source`, `--library-dir`) override `configs/stage1.yaml`, which overrides the built-in `DEFAULTS` in `runtime/config.py`. A missing config file or key is never fatal — defaults are the source of truth.

## Architecture — the load-bearing ideas

**The reduction funnel.** Each stage strips information down toward the one thing on the wire to the generator: a ~768-d **style vector**. Raw perception (skeletons, faces, rPPG) → features → a tiny `StateVector` → **weights over anchor prompts** → blended style vector. A skeleton or heatmap *never* reaches the generator, and the style vector deliberately does **not** carry tempo (tempo is a separate control channel) and is smoothed/rate-limited so the music morphs rather than jumps.

**Types are the seams (`src/entrain/types.py`).** Every module communicates only through these plain dataclasses (`Frame`, `PoseFeatures`, `AffectFeatures`, `CardiacFeatures`, `StateVector`, `AnchorSet`, `StyleVector`, `AudioChunk`). Honoring the type contract is what makes the two big forks one-line swaps: selection↔generation engine, and heuristic↔learned policy. When changing data flow, update the type and the contract table in `docs/architecture.md` together.

**Two decoupled loops (`runtime/loops.py`) — non-negotiable.** A **fast audio loop** that must never stall (read style slot → `engine.next_chunk()` → sink) and a **slow vision loop** (capture → perception → state → policy → write slot). They communicate only through `runtime/shared_state.py:StyleSlot`, an atomically-updated, rate-limited slot — no locks on the audio path. The worst case is *stale steering*, never *silence*. Do not move perception/policy work onto the audio thread.

**MLX thread-affinity.** All MLX work — model load, anchor embedding, generation — must run on the **audio thread**, because MLX streams are thread-local. This is why anchor embedding is deferred into the audio loop's `warmup()` (see `runtime/app.py`) and the vision loop waits on an `anchors_ready` event before it can blend. Keep this invariant if you touch engine startup or anchor embedding.

**The `MusicEngine` Protocol (`generation/engine.py`)** is the de-risking swap point: `SynthEngine` (audible now, cheap local synth), `SelectionEngine` (de-risked real-track path, currently a stub), `Mrt2Engine` (Magenta RealTime 2 streaming). It also exposes `embed_text` because text and audio share one style space (MusicCoCa-style). The `synth`/`selection` engines use the simple blocking `AudioSink`; `mrt2` needs `QueuedAudioSink` (generate-ahead) to stay gap-free.

## Implementation status (important)

Stage 1 is partly stubs. The frozen/heavy boxes — camera, pose, affect, rppg, real engines — may raise `NotImplementedError`. What is genuinely implemented and tested: **anchor blending** (`policy/anchors.py`), the **heuristic policy axis** (`policy/heuristic.py`), **state smoothing** (`state/estimator.py`), and the **rate-limited style slot** (`runtime/shared_state.py`). When adding tests, follow `tests/test_pipeline_seams.py`: test the real logic at the seams; construct fakes via `AnchorBank.__new__` rather than triggering heavy loads.

## Conventions

- Source layout is `src/entrain/`; package discovery and `pythonpath` are configured in `pyproject.toml`.
- Anchor order in `configs/anchors.yaml` is **low-energy → high-energy** and `HeuristicPolicy` depends on that ordering.
- `PoseFeatures.micromotion` is the subtle-movement channel for a near-still seated listener (windowed std of the detrended bob); the estimator fuses it as `max(motion_energy, micro_weight · micromotion)` so gross movement and micro-fidget share one drive. Knobs: `pose.micro_gain`, `state.micro_weight`.
- `responsive: false` in the config freezes the `StyleSlot` after warmup — the static-generative A/B baseline (same loop, steering severed) for testing whether the closed loop beats fixed music.
- Movement thresholds are **per-session calibrated**, not absolute: `state/calibration.py:SignalCalibrator` tracks a resting floor (fast-fall / slow-rise) + spread of motion energy; the vision loop gates the MRT2 drum pulse on `excess ≥ mrt2.drum_gate` and fires accents off `excess / spread` (`mrt2.onset_rise_sigma`). This is what stops drums firing constantly on a still seated bob, and is the normalized input the future learned policy will consume. `OnsetDetector`'s API is signal-agnostic — the calibration happens at the call site, so don't bake absolute-energy assumptions into it.
- Perception runs inside a try/except in the vision loop: a single bad frame must not freeze the music. Preserve that — log once, continue.
- Apple Silicon / MLX is assumed (unified memory is the latency edge). This is a Mac project.
