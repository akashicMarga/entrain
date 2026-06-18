# Architecture — module contracts

This is the engineering companion to `README.md`. It pins down the **data on the wire**
between stages and the responsibility of each module. The design rationale (the science,
the model landscape, the naming) lives in `system-design.md`.

## The reduction funnel

The generator understands exactly one thing: a **style-conditioning vector** (plus its
own past audio). Nothing like a skeleton ever reaches it. Each stage strips information:

```
raw perception   →   features        →   state vector              →   style vector (on the wire)
pose skeletons       motion energy       {arousal, valence,            0.1·A + 0.3·B + 0.6·C
faces                valence/arousal      energy, synchrony, tempo}    = ~768-d blend of anchors
rPPG                 heart rate
```

1. **Raw → features.** Pose aggregates to motion energy / movement tempo / synchrony;
   faces average to valence + arousal; rPPG gives heart rate + trend.
2. **Features → state.** A tiny human-readable summary. Still internal — not sent.
3. **State → signal.** Predefined **anchor prompts** ("calm ambient", "groovy house",
   "peak techno") are each embedded once into the model's shared text/audio space. Crowd
   state becomes **weights** over those anchors; the style vector is the weighted blend.
   *That blended vector is the only thing on the wire.* Stage 1: weights from a rule.
   Stage 3: weights from the learned policy. Output type never changes.

Two things the style vector does **not** carry: **tempo** (use prompt choice / MRT2 MIDI
channel / downstream time-stretch) and **twitch** (it's smoothed + rate-limited so the
music morphs, not jumps).

## Data contracts (`src/entrain/types.py`)

| Type | Produced by | Consumed by | Fields |
|---|---|---|---|
| `Frame` | `capture` | `perception` | `rgb: np.ndarray`, `t: float`, `frame_id: int` |
| `PoseFeatures` | `perception.pose` | `state` | `motion_energy`, `movement_tempo`, `synchrony`, `lead_hand_height`, `micromotion` |
| `AffectFeatures` | `perception.affect` | `state` | `valence`, `arousal`, `n_faces` |
| `CardiacFeatures` | `perception.rppg` | `state` | `heart_rate`, `hr_trend`, `confidence` |
| `StateVector` | `state.estimator` | `policy` | `arousal`, `valence`, `energy`, `synchrony`, `tempo` |
| `AnchorSet` | `policy.anchors` | `policy` | `names: list[str]`, `embeddings: np.ndarray (k×d)` |
| `StyleVector` | `policy` | `generation` | `vec: np.ndarray (d,)`, `weights: dict[str,float]` |
| `AudioChunk` | `generation` | `output` | `pcm: np.ndarray`, `sample_rate: int` |

These types are the seams. Any module can be swapped as long as it honors them — that's
what makes the selection→generation fork and the heuristic→learned-policy fork cheap.

## Module responsibilities

### `capture/` — [world]
`camera.py` owns the OpenCV capture (`cv2.VideoCapture`) at 30 fps. Pushes `Frame`s into a
shared ring buffer. **Disable Center Stage / auto-framing** — it stomps the tiny rPPG
signal. Owns no ML.

### `perception/` — [frozen + one trained box]
- `pose.py` — **frozen.** MediaPipe Pose or YOLO-pose. Skeletons → motion energy, movement
  tempo, cross-subject movement synchrony, and **micromotion** — the subtle-movement
  intensity (windowed std of the detrended bob) that survives when gross motion reads ~0,
  so a seated, still-but-engaged listener still drives the loop (the single-subject desk case).
- `affect.py` — **🪸 TRAINED.** A small MLX head on a pretrained FER backbone →
  valence/arousal. We work from plain RGB (no depth/blendshape shortcuts), so we build our
  own affect read-out; a pretrained face detector (frozen) supplies the crops.
- `rppg.py` — **frozen** classical DSP (POS/CHROM). Contactless heart rate from facial
  color change. Close-range single-face only; degrades hard under motion/low-light/distance.

### `state/` — [build]
- `estimator.py` fuses the three feature streams and **smooths** them into a `StateVector`.
  Movement drive is `max(motion_energy, micro_weight · micromotion)` — a dancer is driven by
  gross motion, a near-still listener by micromotion. The impact metric is **synchrony**
  (inter-subject correlation over ~10 s windows), not average arousal — a loud bad drop spikes
  arousal but not synchrony.
- `calibration.py` — `SignalCalibrator`: per-session adaptive normalization of a movement
  signal. Tracks a **resting floor** (falls fast → finds your quiet level / camera baseline,
  rises slowly → sustained dancing isn't normalized into "rest") plus the typical **spread**
  above it. The runtime gates drums on `excess` ("movement above your rest" in native units)
  and feeds accents `excess / spread` ("a jump relative to your usual"), so thresholds are
  *per-person* instead of fixed constants — the fix for drums firing constantly on a still
  seated bob. Same idea as the estimator's `hr_baseline`, and it's the normalized input the
  learned policy (`policy/learned.py`) would consume — heuristic gates and the future learned
  controller share one calibrated representation.

### `policy/` — [the research]
- `anchors.py` — loads `configs/anchors.yaml`, embeds each anchor prompt once via the
  generator's text encoder (MusicCoCa-style shared space), caches the `AnchorSet`.
- `heuristic.py` — **Stage 1, no training.** Hand-written `StateVector → weights`.
- `learned.py` — **🪸 Stage 3.** MLX net `StateVector → weights`. Trained offline (RL /
  imitation) in `training/policy/`. Same output signature as the heuristic.

Both emit a `StyleVector` = `Σ weightᵢ · anchorᵢ`. The policy is the original contribution;
the right action isn't in any dataset — it's learned from how the crowd responds *after*
you act.

### `generation/` — [frozen renderer]
- `engine.py` — abstract `MusicEngine`: `start()`, `set_style(StyleVector)`, `next_chunk() -> AudioChunk`.
  This interface is the selection↔generation swap point.
- `mrt2.py` — Magenta RealTime 2 streaming adapter (continuous, mid-stream steerable).
- `selection.py` — **de-risk path.** Picks + crossfades tracks from a real library by
  nearest-anchor; reliably musical, no uncanny risk. Same `MusicEngine` interface.

### `output/` — [world]
`audio_sink.py` — `sounddevice` (PortAudio) output stream + ring buffer → speakers. Must
never underrun.

### `runtime/` — [build, the discipline]
- `shared_state.py` — an atomically-updated **slot** holding the current `StyleVector`.
  The audio loop reads it; the vision loop writes it. No locks on the audio path. The
  `responsive=False` config freezes steering after the first write (and suppresses onset/
  melody accents) — the **static-generative A/B baseline**: same loop, steering severed,
  to test whether the closed loop is more engaging than fixed music.
- `loops.py` — two loops:
  - **Audio loop (fast, never stalls):** read slot → `engine.next_chunk()` → `output`.
  - **Vision loop (slow, async):** `capture → perception → state → policy → write slot`.
    Also runs the per-frame `SignalCalibrator` on raw motion energy and, for the MRT2 drum
    channel, fires accents off the calibrated level and **gates the tempo pulse** on
    `excess ≥ drum_gate` (writes tempo 0 below it, handing rhythm back to the model).
- `app.py` — constructs both loops, picks the engine, runs Stage 1.

## Why the two-loop split is non-negotiable

If perception/policy ran on the audio thread, a slow frame or a model hiccup would
underrun the speaker buffer — an audible glitch in front of a crowd. Decoupling means the
worst case is *stale steering* (music keeps playing, just doesn't update for a beat),
never *silence*. Standard real-time-audio discipline.
