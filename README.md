# Entrain

*A music system where the listener is part of the loop.* Entrain senses how people
respond to music — through a camera, and eventually through physiology — and steers
what plays in real time. The system **entrains** to the listener; "entrain" reads as
both *lock rhythms together* and *train a model*, which is exactly what this is.

The full design rationale lives in `docs/system-design.md` (the thesis, the science,
the model landscape). **This README is the build plan.** `docs/architecture.md` is the
module-by-module contract.

---

## The one-line architecture

```
camera → perception → state estimate → control policy → generation → music → crowd → (back to camera)
```

The intelligence lives in **perception** and **policy**. The generator is a frozen
renderer steered by a single ~768-d style vector. Two things never cross the wire to
the generator: a skeleton/heatmap (reduced to a state vector first) and tempo (the
style vector controls character, not BPM).

---

## What's frozen vs. what we train

This is the whole project in one table. We own and train **two boxes**: affect read-out
and control policy. Everything else is pretrained, classical DSP, or "the world."

| Stage | Module | Status | Owner |
|---|---|---|---|
| Camera | `capture/` | **world** (OS/hardware) | OpenCV |
| Pose → movement energy | `perception/pose.py` | **frozen** (pretrained) | MediaPipe / YOLO-pose |
| Face → affect (valence/arousal) | `perception/affect.py` | **TRAINED** 🪸 | us (MLX) |
| rPPG → heart rate | `perception/rppg.py` | **frozen** (classical DSP) | us, but no training |
| Fusion → state vector | `state/estimator.py` | **build** (mostly logic) | us |
| State → style trajectory | `policy/` | **TRAINED** 🪸 (the research) | us (MLX) |
| Generation | `generation/` | **frozen** | MRT2 / ACE-Step / Stable Audio |
| Output | `output/` | **world** | sounddevice |

🪸 = "coral box" — the only two parts with learnable weights. The music generator is
**not** what you train; that's GPU-months and it's solved. The original problem is the
*policy that closes the loop*.

---

## Milestones

### Stage 1 — Zero training. Prove the loop end-to-end on a MacBook.
Glue pretrained pieces with a **hand-written heuristic** mapping (more movement → slide
toward higher-energy anchors). Single subject at a desk.

- [ ] `capture/camera.py` — 30 fps RGB frames into a shared buffer (dev: webcam).
- [ ] `perception/pose.py` — pose → motion energy / movement tempo (frozen).
- [ ] `state/estimator.py` — fuse + smooth into `{arousal, valence, energy, synchrony, tempo}`.
- [ ] `policy/anchors.py` — embed 3 anchor prompts once into the model's style space.
- [ ] `policy/heuristic.py` — state → anchor weights → blended style vector.
- [ ] `runtime/loops.py` — **two decoupled loops** (fast audio, slow vision) + shared slot.
- [ ] `generation/selection.py` **or** `generation/mrt2.py` — render from the style vector.
- [ ] `runtime/app.py` — wires it; `scripts/run_stage1.py` launches it.

**Exit criterion:** moving in front of the camera audibly shifts the music, smoothly,
with no audio dropouts, for 10 minutes straight.

### Stage 2 — Light supervised. Make affect survive real crowd conditions.
- [ ] Collect/label festival footage → `training/affect/`.
- [ ] Adapt `perception/affect.py` read-out (transfer from a pretrained FER backbone).
- [ ] Add rPPG (`perception/rppg.py`, POS/CHROM) for the close-range single-face case.
- [ ] Validate movement-synchrony estimate against the wearable subset.

**Exit criterion:** affect + synchrony estimates correlate with held-out crowd labels.

### Stage 3 — The contribution. Learn the control policy.
- [ ] `policy/learned.py` — MLX policy net: state → anchor weights.
- [ ] `training/policy/` — **offline RL / imitation** from festival footage (music paired
      with crowd reaction). Reward = **cross-crowd synchrony**, not average arousal.
- [ ] Restraint/narrative shaping so the loop doesn't always-ramp-up or oscillate.

**Exit criterion:** the learned policy beats the Stage-1 heuristic on held-out
engagement/synchrony, offline, without live experimentation.

---

## The de-risking fork (read before Stage 1)

The loop **does not have to generate.** A camera signal driving a
**selection-and-mixing** engine over a real track library (`generation/selection.py`) is
lower-risk and reliably musical. Generation (`generation/mrt2.py`) is the novel,
higher-uncanny-risk bet. **Prove the closed loop with selection first**, then swap in
generation once the policy is trustworthy — *same camera, same signal, same policy,
safer output stage.* The `MusicEngine` interface (`generation/engine.py`) makes this a
one-line swap.

---

## Runtime discipline (the make-or-break decision)

**Two decoupled loops.** A fast real-time audio/generation loop that must never stall,
and a slower vision/policy loop that updates a shared "current style vector" slot
asynchronously. The audio loop always reads the freshest vector; the camera never blocks
the sound. See `runtime/shared_state.py` and `runtime/loops.py`.

On Apple Silicon, MLX uses **unified memory** — the whole perceive→decide→generate loop
lives in one shared space with no host↔device copies. That co-residency is the latency
edge and the reason this is a Mac/MLX project.

---

## Layout

```
entrain/
├── README.md                 ← this build plan
├── pyproject.toml
├── configs/
│   ├── anchors.yaml          anchor prompts spanning the musical range
│   └── stage1.yaml           heuristic mapping + smoothing/rate-limit params
├── docs/
│   ├── system-design.md      the full design rationale (the "why")
│   └── architecture.md       module contracts + data-on-the-wire (the "how")
├── src/entrain/
│   ├── types.py              shared data contracts (Frame, StateVector, StyleVector…)
│   ├── capture/              camera frames                         [world]
│   ├── perception/           pose, affect 🪸, rppg                 [frozen + 1 trained]
│   ├── state/                fusion → state vector                 [build]
│   ├── policy/               anchors, heuristic, learned 🪸        [the research]
│   ├── generation/           engine iface, mrt2, selection         [frozen]
│   ├── output/               audio sink                            [world]
│   └── runtime/              two loops, shared slot, app           [build]
├── training/
│   ├── affect/               Stage 2 supervised
│   └── policy/               Stage 3 offline RL / imitation
├── scripts/run_stage1.py
└── tests/
```

## Quickstart (Stage 1)

```bash
pip install -e .
python scripts/run_stage1.py                       # synth engine (no model) + live HUD
python scripts/run_stage1.py --source synthetic --headless   # no camera / no audio device
```

### Generative engine — Magenta RealTime 2 (real streaming music)

MRT2 streams raw audio on Apple Silicon via MLX, steered live by the blended anchor style
vector (the library's own `tokenize(mean(embeddings))` pattern). ~2× real-time for
`mrt2_small` on an M-Pro; `mrt2_base` (2.4B) is the quality upgrade.

```bash
pip install -e ".[mrt2]"        # installs magenta-rt[mlx]
mrt models init                 # MusicCoCa + SpectroStream codec
mrt models download mrt2_small  # streaming model (or mrt2_base)
python scripts/run_stage1.py --engine mrt2
```

Notes: model work (load + anchor embed + generation) all runs on the audio thread — MLX
streams are thread-local — and a generate-ahead `QueuedAudioSink` keeps the stream
gap-free. First start has a ~20 s model-load pause before audio begins. Set `mrt2.size:
mrt2_base` in `configs/stage1.yaml` for higher quality.

The **selection** engine (de-risked real-track path) remains a stub for later.
