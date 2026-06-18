# Entrain — System Design

*A music system where the listener is part of the loop. Entrain senses how people respond
to music — through a camera, and eventually through physiology — and steers what plays in
real time. The name is the spine: the system entrains to the listener, and "entrain"
reading as both "lock rhythms together" and "train a model" is exactly what it is.*

This is the design rationale (the "why"). `README.md` is the build plan; `architecture.md`
is the module contract. The annotated scientific grounding lives in the reading-list files.

---

## 1. The thesis

Most music AI is about *making* music. Entrain pulls the other thread: **how we react to
music.** Not how to synthesize a track, but how a track acts on the person hearing it. The
listener isn't the audience for the system; the listener is *inside* it. Three faces of one
question:

- **Recognition — the memory reaction.** How a few noisy seconds map to "I know this song."
- **Feeling — the affective reaction.** How timbre and texture shape what we feel.
- **Embodiment — the bodily reaction.** A crowd-responsive DJ: sense how a room moves and
  adapt in a real-time perceive → decide → act loop. Applied rhythmic *entrainment*.

The connective tissue across feeling and embodiment is entrainment itself: internal
oscillators lock to the pulse and *predict*; prediction generates anticipation; anticipation
is dopaminergically rewarded. That chain is why temporal structure feels good, and it's the
mechanism Entrain reads and exploits.

---

## 2. Sensing — from camera to an internal energy map

Start with video, add physiology only to understand deeper impact.

**Sensor ladder:**
- **Camera (scalable now):** crowd movement/pose → arousal; facial expression → valence; and
  the same camera yields a contactless cardiac signal via **rPPG**. rPPG degrades hard under
  motion/low-light/distance — close-range per-face only.
- **Wearables (instrumented subset):** wrist HR/HRV and especially EDA. Doesn't scale; you
  don't need everyone.
- **EEG (single-user only):** richest for affect, least practical on a floor.

**The laptop camera** is a single-subject, desk-distance dev sensor: pose (solid, via
MediaPipe/YOLO), face + expression (a pretrained detector finds faces; *affect* needs our
own model on plain RGB), rPPG (classical POS/CHROM, disable Center Stage). Perfect dev rig,
wrong deployment sensor. Algorithms port; the sensor changes.

**The impact metric is synchrony, not average arousal.** Audiences' heart rate, respiration,
skin conductance, and movement synchronize during live music, and the *degree* of synchrony
tracks engagement (Dynamic Attending Theory, Large & Jones). So the reward is **cross-crowd
synchrony** (inter-subject correlation over ~10s windows), not average arousal — which a
loud bad drop spikes.

---

## 3. The closed loop

```
camera → perception → state estimate → control policy → generation → music → crowd → (back to camera)
```

The intelligence lives in perception and policy. The generator is a renderer steered by a
vector.

---

## 4. The control interface — what's on the wire

The generator understands one thing: a **style-conditioning vector** (plus its own past
audio). No skeleton ever reaches it. Funnel: raw perception → features → state vector →
**anchor weights → blended ~768-d style embedding** (the signal). Stage 1 the map is a rule;
Stage 3 it's the learned policy. The embedding does not carry **tempo** and is
**smoothed/rate-limited**.

---

## 5. The generation engine — Magenta RealTime

Not a MIDI model — it generates **raw audio** as a token sequence. Three parts:
**SpectroStream** (neural codec, audio ↔ tokens), **the transformer** (next-token
prediction over audio-codec tokens), **MusicCoCa** (CLIP-like, puts text and audio in one
space). Trained self-supervised on ~190k hours. A text prompt works at inference because
MusicCoCa aligns text and audio — the model just sees a point in style-space.

---

## 6. Model landscape

The fork: **streaming autoregressive vs diffusion clip.** Streaming (Magenta RealTime 2)
steers mid-stream while it plays — what the loop needs. Diffusion clip (ACE-Step / Stable
Audio) renders fast *clips* you can only re-prompt between. The trap: fast clip generation
is not steerable streaming.

**Dual-engine division of labor:** MRT2-style streaming for the live steerable bed; a
diffusion model for on-device clips, ambience, full songs with vocals, and LoRA
personalization (wrap live use in generate-ahead + crossfade).

---

## 7. Do we train anything?

Almost none of the heavy parts. **You do not train a music generator.** Frozen: pose, rPPG
(classical DSP), the generator. **Trained (coral boxes):** facial affect, fusion/state, and
the real one — the **control policy.** Three stages: (1) zero training, heuristic glue;
(2) light supervised affect adaptation; (3) the contribution — learn the closed-loop policy
via offline RL / imitation from festival footage. The hard original problem is the policy
that closes the loop and its reward (synchrony).

---

## 8. Feasibility

**Solid:** camera → control representation → conditioned music model is a real, demonstrated
pattern (V2Meow, Foley Music, biosignal→music). The plumbing isn't in question.

**The genuine unknowns (all downstream of the policy):** the policy is unproven; closed
loops can chase their own tail (oscillate/lag/always-ramp-up); real-scene signal is thinner
than the desk demo (movement survives, affect/rPPG largely don't at distance);
granularity/latency (style controls vibe/energy, not tempo/beat).

**De-risking fork:** the loop doesn't have to *generate*. A camera signal driving
**selection-and-mixing** over a real track library is lower-risk and reliably musical. Prove
the closed loop with selection first, then swap in generation — same camera, signal, policy.

---

## 9. MLX / on-device runtime

On Apple Silicon, MLX uses **unified memory** — the whole loop lives in one shared space,
no host↔device copies; sensing and generation co-resident on one chip. The make-or-break
decision: **two decoupled loops** — a fast audio/generation loop that never stalls, and a
slower vision/policy loop updating a shared "current style vector" slot asynchronously.

---

## 10. Where it points

Non-Western MIR (raga/microtonality, LoRA-personalize on Indic material);
neuro-adaptive single-user biosignal → adaptive music (sleep/focus/rehab); the recognition
pillar (fingerprinting inside DJ mixes).

---

## 11. Naming

**Entrain** — the scientific spine (entrainment), repo-clean, and the pun
(entrainment / training) fits an on-device trainable system about entrainment.

---

## 12. Grounding

Load-bearing anchors (full sets in the reading-list files):
Large & Jones (1999); Nozaradan et al. (2011); Trost et al. (2014/2024); Juslin &
Västfjäll (2008); Koelsch (2014); Salimpoor et al. (2011); Krumhansl (2010); V2Meow; Foley
Music (Gan et al. 2020); Ehrlich et al. (2019); MERT; EnCodec; DDSP; MusicGen; Magenta
RealTime; ACE-Step; Stable Audio 3.0.
