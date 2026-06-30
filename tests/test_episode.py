"""EpisodeBuffer: the loop's memory — state trajectory + action log + lagged responses.

Pure bookkeeping, tested with synthetic state streams (no camera/model). The point is to
pair "what we did" with "what the room did next", which is both the director's cause->effect
read and the offline training tuples.
"""

import numpy as np

from entrain.runtime.episode import EpisodeBuffer
from entrain.types import Directive, AnchorSpec, StateVector


def _state(energy=0.0, synchrony=0.0, valence=0.0):
    return StateVector(arousal=energy, valence=valence, energy=energy,
                       synchrony=synchrony, tempo=120.0)


def test_window_summary_means_and_trend():
    buf = EpisodeBuffer(fps=10, history_s=60)
    for i in range(50):                       # 5 s at 10 fps, energy ramps 0 -> ~0.5
        buf.add_state(t=i * 0.1, state=_state(energy=i * 0.01, synchrony=0.2))
    ctx = buf.director_context(now=4.9, window_s=20.0)
    assert ctx.energy_trend > 0               # rising energy -> positive slope
    assert abs(ctx.synchrony_mean - 0.2) < 1e-6
    assert ctx.elapsed_s == 4.9               # now - t0 (first sample at t=0)


def test_transition_captures_response_after_lag():
    buf = EpisodeBuffer(fps=10, history_s=60)
    # synchrony is low at/around the action, then rises afterward -> the move "worked".
    for i in range(40):
        t = i * 0.1
        sync = 0.1 if t < 2.5 else 0.8
        buf.add_state(t=t, state=_state(energy=0.5, synchrony=sync))
    buf.add_action(t=2.0, weights={"peak": 1.0})
    trans = buf.transitions(lag_s=1.0, now=3.9)
    assert len(trans) == 1
    assert trans[0].response_synchrony > 0.5  # synchrony rose over the lag window
    assert trans[0].action.weights == {"peak": 1.0}


def test_fresh_action_is_not_scored_before_its_lag_elapses():
    buf = EpisodeBuffer(fps=10, history_s=60)
    for i in range(30):
        buf.add_state(t=i * 0.1, state=_state(energy=0.5, synchrony=0.3))
    buf.add_action(t=2.8, weights={"groove": 1.0})   # only 0.1 s before 'now'
    assert buf.transitions(lag_s=1.0, now=2.9) == []  # too recent to have a response yet


def test_last_transition_is_the_most_recent_scorable():
    buf = EpisodeBuffer(fps=10, history_s=60)
    for i in range(60):
        buf.add_state(t=i * 0.1, state=_state(energy=0.5, synchrony=0.3))
    buf.add_action(t=1.0, weights={"calm": 1.0})
    buf.add_action(t=3.0, weights={"peak": 1.0})
    last = buf.last_transition(lag_s=1.0, now=5.9)
    assert last is not None and last.action.weights == {"peak": 1.0}


def test_actions_can_log_a_directive():
    buf = EpisodeBuffer(fps=10, history_s=60)
    for i in range(30):
        buf.add_state(t=i * 0.1, state=_state(energy=0.4, synchrony=0.2))
    d = Directive(anchors=[AnchorSpec("ambient", "warm beatless pads")], intent="cool down")
    buf.add_action(t=1.0, directive=d)
    trans = buf.transitions(lag_s=1.0, now=2.9)
    assert trans[0].action.directive.intent == "cool down"


def test_director_context_reports_last_synchrony_response():
    """director_context surfaces Δsynchrony after the last action when given a feedback lag —
    the 'did my last move work?' signal the HeuristicDirector hill-climbs on."""
    buf = EpisodeBuffer(fps=10, history_s=60)
    for i in range(40):
        t = i * 0.1
        sync = 0.1 if t < 2.5 else 0.7
        buf.add_state(t=t, state=_state(energy=0.5, synchrony=sync))
    buf.add_action(t=2.0, directive=Directive(anchors=[AnchorSpec("peak", "p")]))
    ctx = buf.director_context(now=3.9, feedback_lag_s=1.0)
    assert ctx.last_response_synchrony > 0.5             # the move raised synchrony
    assert buf.director_context(now=3.9).last_response_synchrony == 0.0   # off without a lag


def test_snapshot_capture_keeps_frame_optional():
    buf = EpisodeBuffer()
    buf.snapshot(t=1.0, state=_state(energy=0.3), label="before")
    buf.snapshot(t=4.0, state=_state(energy=0.7), label="after",
                 frame=np.zeros((4, 4, 3), dtype=np.uint8))
    snaps = buf.snapshots
    assert [s.label for s in snaps] == ["before", "after"]
    assert snaps[0].frame is None and snaps[1].frame is not None


def test_empty_buffer_is_safe():
    buf = EpisodeBuffer()
    ctx = buf.director_context(now=0.0)
    assert ctx.energy_mean == 0.0 and ctx.elapsed_s == 0.0
    assert buf.transitions(lag_s=1.0) == []
    assert buf.last_transition(lag_s=1.0) is None
