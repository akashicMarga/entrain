"""SignalCalibrator: the per-session movement calibration that gates drums/accents.

Tested with synthetic signal streams so it needs no camera or model. The point of the
class is that thresholds become "above YOUR resting level" instead of absolute constants.
"""

from entrain.state.calibration import SignalCalibrator


def _feed(cal, signal):
    for v in signal:
        cal.update(v)


def test_floor_learns_resting_level_and_excess_is_zero_at_rest():
    cal = SignalCalibrator()
    _feed(cal, [0.1] * 100)                 # sit quietly at a 0.1 baseline
    assert abs(cal.floor - 0.1) < 0.02      # floor settles on the resting level
    assert cal.excess(0.1) == 0.0           # being at rest is not "movement"


def test_excess_rises_for_real_movement_above_rest():
    cal = SignalCalibrator()
    _feed(cal, [0.1] * 100)                 # rest at 0.1
    # a sudden move to 0.5: the floor rises only slowly, so most of it reads as excess.
    assert cal.excess(0.5) > 0.35


def test_sustained_movement_stays_above_the_gate():
    """A dancer shouldn't get normalized into 'rest' — the floor rises slowly."""
    cal = SignalCalibrator()
    _feed(cal, [0.1] * 100)                 # rest
    _feed(cal, [0.5] * 200)                 # then dance for a while
    assert cal.excess(0.5) > 0.1            # still clears the default drum_gate (0.1)


def test_floor_falls_fast_when_movement_stops():
    cal = SignalCalibrator()
    _feed(cal, [0.1] * 50)                  # rest
    _feed(cal, [0.5] * 100)                 # move (raises the floor a little)
    _feed(cal, [0.1] * 40)                  # stop -> floor should drop back fast
    assert cal.excess(0.1) == 0.0           # quiet again reads as rest within ~1s


def test_spread_never_below_floor():
    cal = SignalCalibrator(spread_floor=0.02)
    _feed(cal, [0.3] * 100)                 # perfectly steady -> excess ~0 -> spread decays
    assert cal.spread >= 0.02               # but is clamped, so onset levels stay finite


def test_no_history_is_safe():
    cal = SignalCalibrator()
    assert cal.excess(0.9) == 0.0           # before any update, nothing is "movement"
    assert cal.floor == 0.0
