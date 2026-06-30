"""The loop's memory. [BUILD — bookkeeping, not a model]

A single frame is a *state*; understanding "did the move I just made work?" needs the
room's trajectory AND a log of what we did, paired with a lag. That pairing is the whole
closed-loop thesis (how the crowd responds *after* you act) — and it's cheap: we already
produce a `StateVector` every frame, we just have to remember it alongside the actions.

`EpisodeBuffer` holds two bounded streams:

    states:  ring of (t, StateVector)          — the room over time
    actions: log of (t, weights / Directive)   — what the policy/director did

and answers the two questions the rest of the system asks:

  * `director_context(now)` — the slow, aggregated view (means/trends) the Director reasons
    over (fills `DirectorContext`).
  * `transitions(lag_s)` — for each action, the room's response over the next `lag_s`: the
    "what changed when I tuned the music" read for the director, and the `(state, action,
    response)` tuples that TRAIN the learned policy/director offline. The headline response
    is the change in SYNCHRONY (engagement), not average arousal.

Optional `snapshot()` captures a frame + state at an event (e.g. a Directive change) for a
VLM to compare before/after — cheaper than streaming video into a model.

NOT wired into the hot loop yet: nothing consumes it until the Director/learned policy land,
so the writes (`add_state` / `add_action` per frame, `snapshot` on Directive change) would
be dead weight. Plugs into `VisionLoop` — see the call sites noted there when enabling it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from entrain.types import Directive, DirectorContext, StateVector


@dataclass(slots=True)
class ActionRecord:
    """Something the system did, timestamped. [runtime -> episode]

    A fast-policy style move carries `weights`; a slow Director re-author carries a
    `Directive` (the anchor set changed). Either may be present.
    """

    t: float
    weights: dict[str, float] | None = None
    directive: Directive | None = None


@dataclass(slots=True)
class Snapshot:
    """A frame + state captured at an event, for before/after VLM comparison."""

    t: float
    state: StateVector
    label: str = ""                       # "before" / "after" / the Directive intent
    frame: np.ndarray | None = None       # optional image; kept out unless a VLM needs it


@dataclass(slots=True)
class Transition:
    """One (state, action, response) tuple: what the room did in the `lag_s` after an action.

    The director's cause->effect read AND the offline training datum. `response_synchrony`
    is the headline reward proxy (engagement), deliberately NOT average arousal — a loud bad
    drop spikes arousal but not synchrony. [episode -> policy.director / training]
    """

    t: float
    action: ActionRecord
    state_before: StateVector
    state_after: StateVector
    lag_s: float

    @property
    def response_synchrony(self) -> float:
        return self.state_after.synchrony - self.state_before.synchrony

    @property
    def response_energy(self) -> float:
        return self.state_after.energy - self.state_before.energy

    @property
    def response_valence(self) -> float:
        return self.state_after.valence - self.state_before.valence


class EpisodeBuffer:
    """Bounded memory of the perceive->act loop: state trajectory + action log."""

    def __init__(self, fps: int = 30, history_s: float = 120.0,
                 max_actions: int = 512, max_snapshots: int = 64) -> None:
        self.t0: float | None = None      # set on the first sample; elapsed_s is t - t0
        self._states: deque[tuple[float, StateVector]] = deque(
            maxlen=max(8, int(fps * history_s))
        )
        self._actions: deque[ActionRecord] = deque(maxlen=max_actions)
        self._snapshots: deque[Snapshot] = deque(maxlen=max_snapshots)

    # --- writers (called by the runtime) ----------------------------------------

    def add_state(self, t: float, state: StateVector) -> None:
        if self.t0 is None:
            self.t0 = t
        self._states.append((t, state))

    def add_action(self, t: float, weights: dict[str, float] | None = None,
                   directive: Directive | None = None) -> None:
        self._actions.append(ActionRecord(t=t, weights=weights, directive=directive))

    def snapshot(self, t: float, state: StateVector, label: str = "",
                 frame: np.ndarray | None = None) -> None:
        self._snapshots.append(Snapshot(t=t, state=state, label=label, frame=frame))

    # --- readers (the seams out) ------------------------------------------------

    def director_context(self, now: float, seconds_in_directive: float = 0.0,
                         instruction: str = "", window_s: float = 20.0,
                         feedback_lag_s: float = 0.0) -> DirectorContext:
        """The slow aggregated view the Director reasons over, over the last `window_s`.

        If `feedback_lag_s > 0`, also reports the synchrony response to the director's most
        recent scorable action (`last_transition`) — the "did my last move work?" signal.
        """
        win = [(t, s) for (t, s) in self._states if now - t <= window_s]
        if win:
            ts = np.array([t for t, _ in win], dtype=np.float64)
            energy = np.array([s.energy for _, s in win], dtype=np.float64)
            valence = np.array([s.valence for _, s in win], dtype=np.float64)
            synchrony = np.array([s.synchrony for _, s in win], dtype=np.float64)
            energy_mean = float(energy.mean())
            valence_mean = float(valence.mean())
            synchrony_mean = float(synchrony.mean())
            # signed slope over the window (per second); 0 if too few / no time spread.
            energy_trend = (
                float(np.polyfit(ts - ts[0], energy, 1)[0])
                if len(ts) >= 2 and ts[-1] > ts[0] else 0.0
            )
        else:
            energy_mean = valence_mean = synchrony_mean = energy_trend = 0.0
        last_resp = 0.0
        if feedback_lag_s > 0:
            tr = self.last_transition(feedback_lag_s, now=now)
            if tr is not None:
                last_resp = tr.response_synchrony
        return DirectorContext(
            elapsed_s=(now - self.t0) if self.t0 is not None else 0.0,
            energy_mean=energy_mean, energy_trend=energy_trend,
            valence_mean=valence_mean, synchrony_mean=synchrony_mean,
            seconds_in_directive=seconds_in_directive, instruction=instruction,
            last_response_synchrony=last_resp,
        )

    def transitions(self, lag_s: float, now: float | None = None) -> list[Transition]:
        """For each action, pair the state at action time with the state ~`lag_s` later.

        Only actions old enough to HAVE a post-lag observation are returned, so a fresh
        action isn't scored before its effect could land. These are the training tuples
        and the director's "what changed when I tuned the music" read.
        """
        if not self._states:
            return []
        latest_t = self._states[-1][0] if now is None else now
        out: list[Transition] = []
        for a in self._actions:
            if latest_t - a.t < lag_s:
                continue                       # too recent — effect hasn't had time to land
            before = self._state_at(a.t)
            after = self._state_at(a.t + lag_s)
            if before is None or after is None:
                continue
            out.append(Transition(t=a.t, action=a, state_before=before,
                                   state_after=after, lag_s=lag_s))
        return out

    def last_transition(self, lag_s: float, now: float | None = None) -> Transition | None:
        """The most recent scorable action's response — the director's immediate read."""
        ts = self.transitions(lag_s, now=now)
        return ts[-1] if ts else None

    @property
    def snapshots(self) -> list[Snapshot]:
        return list(self._snapshots)

    # --- internals --------------------------------------------------------------

    def _state_at(self, t: float) -> StateVector | None:
        """The sample whose timestamp is nearest `t` (None if no states recorded)."""
        if not self._states:
            return None
        best = min(self._states, key=lambda ts: abs(ts[0] - t))
        return best[1]
