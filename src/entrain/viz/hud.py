"""Live HUD. [dev tool — see what the loop is doing, read stats]

Draws the camera frame with a stats overlay: motion energy, the state vector, the anchor
weights, the synth params the audio loop is actually rendering, FPS, and a scrolling
energy history. Runs on the MAIN thread (cv2 GUI requirement on macOS); the audio loop
stays on its own background thread, so the window never blocks the sound.

This is a window onto the loop, not part of it — disable with --no-show / headless and
nothing else changes.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from entrain.generation.synth import SynthEngine
from entrain.types import Frame, StateVector, StyleVector

_PANEL_W = 300
_BG = (18, 18, 18)
_FG = (235, 235, 235)
_DIM = (150, 150, 150)
_ACCENT = (90, 200, 90)
_BAR_BG = (55, 55, 55)
# per-anchor bar colour (BGR): calm blue, groove amber, peak red
_ANCHOR_COLOR = {"calm": (200, 140, 60), "groove": (60, 190, 220), "peak": (70, 70, 230)}
_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _note_name(pitch: int) -> str:
    """MIDI pitch -> note name, e.g. 48 -> 'C3', 60 -> 'C4'."""
    return f"{_NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


class Hud:
    """A cv2 window showing the camera + live stats. Returns False to quit."""

    def __init__(self, history: int = 240, window: str = "Entrain — Stage 1",
                 slot=None) -> None:
        self.window = window
        self.slot = slot                # read drum hits from here (optional)
        self._energy_hist: deque[float] = deque(maxlen=history)
        self._t_prev = time.monotonic()
        self._fps = 0.0
        self._drum_flash = 0            # render frames remaining to show a hit
        self._opened = False

    def render(self, frame: Frame, state: StateVector, style: StyleVector) -> bool:
        import cv2

        if not self._opened:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            self._opened = True

        now = time.monotonic()
        dt = now - self._t_prev
        self._t_prev = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)

        # Drum hits fire on the audio thread; latch via the slot so we never miss one.
        if self.slot is not None and self.slot.take_drum_hit():
            self._drum_flash = 5
        elif self._drum_flash > 0:
            self._drum_flash -= 1

        # Display weights from the ACTUAL (rate-limited) style vector, so the bars match
        # the sound, not the policy's instantaneous target.
        w = np.asarray(style.vec, dtype=np.float32)
        s = float(w.sum())
        w = w / s if s > 0 else w
        weights = {n: float(w[i]) for i, n in enumerate(SynthEngine.ANCHOR_ORDER)}
        freq, bright, dens = SynthEngine.params_from_weights(style.vec)

        cam = np.ascontiguousarray(frame.rgb[..., ::-1])      # RGB -> BGR
        h = cam.shape[0]
        panel = np.full((h, _PANEL_W, 3), _BG, dtype=np.uint8)
        self._energy_hist.append(state.energy)
        self._draw_panel(cv2, panel, state, weights, freq, bright, dens)

        canvas = np.hstack([cam, panel])
        cv2.imshow(self.window, canvas)
        key = cv2.waitKey(1) & 0xFF
        return key not in (27, ord("q"))                      # ESC or q -> quit

    def close(self) -> None:
        if self._opened:
            import cv2

            cv2.destroyWindow(self.window)
            self._opened = False

    # --- drawing ----------------------------------------------------------------

    def _draw_panel(self, cv2, panel, state, weights, freq, bright, dens) -> None:
        x = 16
        y = 34
        cv2.putText(panel, "ENTRAIN", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _ACCENT, 2)
        cv2.putText(panel, f"{self._fps:4.1f} fps", (_PANEL_W - 92, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _DIM, 1)
        y += 30

        # Drum indicator: a dot that flashes bright on each hit, dim outline otherwise.
        cx, cy = _PANEL_W - 28, y + 2
        if self._drum_flash > 0:
            cv2.circle(panel, (cx, cy), 11, (80, 90, 250), -1)   # filled flash
            cv2.putText(panel, "HIT", (x, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (80, 90, 250), 2)
        else:
            cv2.circle(panel, (cx, cy), 11, _BAR_BG, 2)          # dim ring
            cv2.putText(panel, "drum", (x, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _DIM, 1)
        y += 26

        y = self._bar(cv2, panel, x, y, "motion energy", state.energy, _ACCENT)
        y += 6
        cv2.putText(panel, "anchor weights", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _DIM, 1)
        y += 20
        for name in SynthEngine.ANCHOR_ORDER:
            y = self._bar(cv2, panel, x, y, name, weights.get(name, 0.0),
                          _ANCHOR_COLOR[name])
        y += 10

        cv2.putText(panel, "synth", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _DIM, 1)
        y += 22
        for label, val in (("freq", f"{freq:6.1f} Hz"),
                           ("bright", f"{bright:.2f}"),
                           ("pulse", f"{dens:.2f}")):
            cv2.putText(panel, f"{label:<7}{val}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _FG, 1)
            y += 24
        y += 6

        y = self._melody(cv2, panel, x, y)

        tempo = f"{state.tempo:5.1f} bpm" if state.tempo > 1 else "  --  "
        hr = f"{state.heart_rate:4.0f} bpm" if state.heart_rate > 1 else "  --  "
        cv2.putText(panel, f"tempo {tempo}   sync {state.synchrony:.2f}",
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _DIM, 1)
        y += 16
        cv2.putText(panel, f"arousal {state.arousal:.2f}   valence {state.valence:+.2f}",
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _DIM, 1)
        y += 16
        cv2.putText(panel, f"heart {hr}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _DIM, 1)
        y += 18

        # Director (slow planner): what it's doing now + the synchrony feedback it acted on.
        intent = self.slot.directive_intent if self.slot is not None else ""
        if intent:
            resp = self.slot.last_response if self.slot is not None else 0.0
            rc = _ACCENT if resp > 0.02 else ((70, 70, 230) if resp < -0.02 else _DIM)
            cv2.putText(panel, f"director  {intent}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (205, 180, 255), 1)
            y += 16
            cv2.putText(panel, f"  dsync {resp:+.2f}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, rc, 1)
            y += 18

        self._sparkline(cv2, panel, x, panel.shape[0] - 54, _PANEL_W - 2 * x, 44)
        cv2.putText(panel, "q / esc to quit", (x, panel.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _DIM, 1)

    def _bar(self, cv2, panel, x, y, label, value, color) -> int:
        cv2.putText(panel, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _FG, 1)
        bw, bh = _PANEL_W - 2 * x, 12
        cv2.rectangle(panel, (x, y + 4), (x + bw, y + 4 + bh), _BAR_BG, -1)
        fill = int(bw * float(np.clip(value, 0.0, 1.0)))
        cv2.rectangle(panel, (x, y + 4), (x + fill, y + 4 + bh), color, -1)
        cv2.putText(panel, f"{value:.2f}", (x + bw - 38, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _FG, 1)
        return y + 4 + bh + 16

    def _melody(self, cv2, panel, x, y) -> int:
        """Lead-note readout: note name + a pitch bar, lit while a note is playing."""
        pitch = self.slot.melody if self.slot is not None else None
        playing = pitch is not None
        color = (210, 230, 90) if playing else _DIM      # cyan-ish when playing
        name = _note_name(pitch) if playing else "--"
        cv2.putText(panel, "melody", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _DIM, 1)
        cv2.putText(panel, name, (x + 78, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,
                    2 if playing else 1)
        y += 12
        # pitch bar: marker position over a display range of MIDI 48..84 (3 octaves)
        bw, bh = _PANEL_W - 2 * x, 10
        cv2.rectangle(panel, (x, y), (x + bw, y + bh), _BAR_BG, -1)
        if playing:
            pos = float(np.clip((pitch - 48) / 36.0, 0.0, 1.0))
            mx = x + int(pos * bw)
            cv2.rectangle(panel, (mx - 3, y - 2), (mx + 3, y + bh + 2), color, -1)
        return y + bh + 18

    def _sparkline(self, cv2, panel, x, y, w, h) -> None:
        cv2.rectangle(panel, (x, y), (x + w, y + h), _BAR_BG, 1)
        cv2.putText(panel, "energy history", (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, _DIM, 1)
        hist = list(self._energy_hist)
        if len(hist) < 2:
            return
        xs = np.linspace(x + 1, x + w - 1, len(hist))
        ys = y + h - 1 - np.clip(hist, 0, 1) * (h - 2)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(panel, [pts], False, _ACCENT, 1)
