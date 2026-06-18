"""Hand height -> melodic pitch. [BUILD — no training]

Maps the lead hand's height to a MIDI pitch so you play a lead line with your body:
raise your hand to go up the scale, lower it to come down, drop it below a threshold to
stop playing. Pitches are quantised to a pentatonic scale, so whatever you "play" stays
consonant with the generated track.

Returns a MIDI pitch (int) or None (hand down / not playing). The engine turns that into
MRT2's 128-slot notes conditioning and decides onset vs. sustain.
"""

from __future__ import annotations

# Major pentatonic scale degrees (semitone offsets). Hard to hit a "wrong" note.
_PENTATONIC = (0, 2, 4, 7, 9)


class MelodyMapper:
    """Quantises hand height to a pentatonic MIDI pitch, with an 'is playing' gate."""

    def __init__(self, root: int = 48, octaves: int = 2, active_threshold: float = 0.35) -> None:
        self.root = root                       # lowest pitch (MIDI), e.g. 48 = C3
        self.octaves = octaves                 # range spanned by the hand
        self.active_threshold = active_threshold  # hand must be above this to play

    def __call__(self, hand_height: float) -> int | None:
        if hand_height < self.active_threshold:
            return None                        # hand down -> silence
        span = self.octaves * len(_PENTATONIC)
        norm = (hand_height - self.active_threshold) / (1.0 - self.active_threshold)
        idx = min(span - 1, max(0, int(norm * span)))
        octave, degree = divmod(idx, len(_PENTATONIC))
        return self.root + 12 * octave + _PENTATONIC[degree]
