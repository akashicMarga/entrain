"""The slow DIRECTOR layer. [the planner above the fast policy]

Two layers steer the music, at two timescales:

  * Director (this file) — SLOW (seconds). Language in, anchor PROMPTS + arc out. Authors
    *which* anchors define the current musical range. This is where a small instruction-
    tuned LLM fits: text is its native domain, the cadence is slow enough that KV-cache
    append is fine, and it never touches the audio/vision control loop. Natural-language
    control ("less aggressive, more groove") and the music's CHARACTER live here — including
    drums, which come from the prompts the Director writes, not from the fast controller.

  * Policy (policy/base.py) — FAST (per frame). state -> weights over the CURRENT anchors.
    A tiny MLX net (or the Stage-1 heuristic). Numbers in, weights out, microseconds.

The seam between them is `Directive` (the authored anchor set + intent). The Director emits
a new one every few seconds; the AnchorBank re-authors + re-embeds (on the generation
thread, like the initial warmup); the Policy keeps weighting over whatever anchors are
current. An LLM in the *fast* loop would be the wrong tool (latency, numeric inputs,
stochastic) — so it lives strictly up here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from entrain.types import AnchorSpec, Directive, DirectorContext


class Director(Protocol):
    """Slow planner: authors the anchor set the fast policy then weights over."""

    def initial(self) -> Directive:
        """The opening anchor set, before any room data."""
        ...

    def revise(self, ctx: DirectorContext) -> Directive | None:
        """Slow tick (every few seconds). Return a new `Directive` to re-author the anchors,
        or `None` to keep the current one. Must be cheap to call and safe to call often;
        the implementation rate-limits its own heavy work (e.g. the LLM call)."""
        ...


class StaticDirector:
    """Stage-1 behavior through the Director seam: the fixed anchors from
    `configs/anchors.yaml`, never revised. Lets the two-layer wiring be exercised and
    tested before any LLM exists — and is the always-available fallback for `LlmDirector`.
    """

    def __init__(self, config_path: str = "configs/anchors.yaml") -> None:
        import yaml

        prompts = yaml.safe_load(Path(config_path).read_text())["anchors"]
        self._directive = Directive(
            anchors=[AnchorSpec(name=p["name"], prompt=p["prompt"]) for p in prompts],
            intent="static stage-1 anchors",
        )

    def initial(self) -> Directive:
        return self._directive

    def revise(self, ctx: DirectorContext) -> Directive | None:
        return None      # fixed set — never changes


class HeuristicDirector:
    """A revising Director with NO model — the "basic version with a director".

    Picks a MODE (a stylistic modifier over the base anchors) from the slow room context and
    re-authors only when the mode changes. Demonstrates the planning layer, including the
    thing a reactive controller can't do: RESTRAINT — it won't ride a build forever, it eases
    off after `sustain_cap_s`. Keeps the base anchor NAMES (calm/groove/peak), appending a
    suffix to each prompt, so (a) the fast policy is unaffected and (b) the synth's keyword
    embedder still matches. The character shift is what re-steers the generator's timbre/drums.

    Same `Director` Protocol as Static/Llm — swap by config (`director: heuristic`).
    """

    # mode -> (suffix appended to every base prompt, intent label)
    _MODES = {
        "neutral": ("", "neutral groove"),
        "lift": (", brighter, more driving, building energy", "building"),
        "cool": (", warmer, softer, more spacious, easing off", "cooling down"),
    }

    def __init__(self, config_path: str = "configs/anchors.yaml",
                 trend_up: float = 0.004, trend_down: float = -0.004,
                 sustain_cap_s: float = 45.0) -> None:
        import yaml

        prompts = yaml.safe_load(Path(config_path).read_text())["anchors"]
        self._base = [(p["name"], p["prompt"]) for p in prompts]   # ordered low -> high
        self.trend_up = trend_up            # energy slope (/s) that counts as "rising"
        self.trend_down = trend_down        # ... and as "falling"
        self.sustain_cap_s = sustain_cap_s  # restraint: max time to hold a build before easing
        self._mode = "neutral"
        self._mode_since = 0.0

    def _directive_for(self, mode: str) -> Directive:
        suffix, intent = self._MODES[mode]
        return Directive(
            anchors=[AnchorSpec(name=n, prompt=p + suffix) for n, p in self._base],
            intent=intent,
        )

    def initial(self) -> Directive:
        return self._directive_for(self._mode)

    def _choose(self, ctx: DirectorContext) -> str:
        held = ctx.elapsed_s - self._mode_since
        # RESTRAINT first: don't ride a build forever -> ease off once held too long.
        if self._mode == "lift" and held >= self.sustain_cap_s:
            return "cool"
        if ctx.energy_trend >= self.trend_up:
            return "lift"
        if ctx.energy_trend <= self.trend_down:
            return "cool"
        # settled + flat after a cooldown -> drift back to neutral
        if self._mode == "cool" and held >= self.sustain_cap_s:
            return "neutral"
        return self._mode

    def revise(self, ctx: DirectorContext) -> Directive | None:
        desired = self._choose(ctx)
        if desired == self._mode:
            return None                     # unchanged -> skip the re-embed
        self._mode = desired
        self._mode_since = ctx.elapsed_s
        return self._directive_for(desired)


class LlmDirector:
    """🪸 The planner: a small instruction-tuned LLM that AUTHORS anchor prompts + arc from a
    language view of the room. Runs every `revise_every_s` (NOT in the control loop).

    Why an LLM fits HERE and not in the fast loop: it works in text, the cadence is slow
    (KV-cache append over a sliding context is fine — it does not need true streaming input),
    and prompt-authoring is exactly what LLMs are good at. Finetuning a small instruction
    model pays off at this layer. Falls back to `StaticDirector` when no model is loaded.
    """

    def __init__(self, model=None, revise_every_s: float = 8.0,
                 fallback: Director | None = None) -> None:
        self.model = model                       # the (small) instruction-tuned LLM
        self.revise_every_s = revise_every_s
        self.fallback = fallback or StaticDirector()
        self._last_revise_s = -1e9

    def initial(self) -> Directive:
        if self.model is None:
            return self.fallback.initial()
        raise NotImplementedError(
            "Seed anchors from the LLM (system prompt -> initial anchors low->high + intent)."
        )

    def revise(self, ctx: DirectorContext) -> Directive | None:
        # Rate-limit the heavy call to the slow cadence; return None between ticks.
        if ctx.elapsed_s - self._last_revise_s < self.revise_every_s:
            return None
        self._last_revise_s = ctx.elapsed_s
        if self.model is None:
            return self.fallback.revise(ctx)
        raise NotImplementedError(
            "Format ctx (+ ctx.instruction) as a prompt; LLM returns anchors low->high + "
            "intent; return None when unchanged so the AnchorBank skips a re-embed."
        )
