"""Audio output. [WORLD — must never underrun]

`AudioSink` is a `sounddevice` (PortAudio) blocking output stream -> speakers -> crowd ->
camera (closing the loop). It sits on the fast audio loop. For Stage 1 we use a *blocking*
write: `stream.write()` returns only once the device has consumed the block, which paces
the audio loop to real time without a separate ring-buffer thread. (The ring-buffer /
callback design — strictly underrun-proof — is the later hardening step.)

`NullSink` consumes chunks and just sleeps for their duration: it lets the whole loop run
headless (no audio device) at the right pace, for tests and CI.
"""

from __future__ import annotations

import queue
import time

import numpy as np

from entrain.types import AudioChunk


class AudioSink:
    """Blocking `sounddevice` output stream."""

    def __init__(self, sample_rate: int = 48_000, channels: int = 2) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream = None

    def open(self) -> None:
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate, channels=self.channels, dtype="float32"
        )
        self._stream.start()

    def write(self, chunk: AudioChunk) -> None:
        if self._stream is None:
            self.open()
        self._stream.write(np.ascontiguousarray(chunk.pcm, dtype=np.float32))  # blocks

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class QueuedAudioSink:
    """Callback-driven `sounddevice` sink with a generate-AHEAD queue.

    A background audio callback pulls continuously from a thread-safe queue while the
    producer (the audio loop) fills it ahead. `write()` blocks only when the queue is full
    (back-pressure paces the producer), and the callback never stalls mid-chunk. This is
    what keeps GENERATED audio gap-free: model inference for the next chunk overlaps
    playback of the current one, instead of leaving a silent hole each cycle.

    Use this for streaming generation (MRT2). For a cheap local synth, the blocking
    `AudioSink` is fine.
    """

    def __init__(self, sample_rate: int = 48_000, channels: int = 2,
                 max_queue: int = 2, blocksize: int = 1024) -> None:
        # max_queue is the generate-AHEAD depth. Keep it SMALL: every queued chunk is
        # already-committed audio that delays the listener hearing a steer. Generation is
        # ~2x real-time, so 2 chunks is enough to avoid underruns while staying responsive.
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._buf = np.zeros((0, channels), dtype=np.float32)
        self._stream = None

    def open(self) -> None:
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate, channels=self.channels, dtype="float32",
            blocksize=self.blocksize, callback=self._callback,
        )
        self._stream.start()

    def _callback(self, outdata, frames, time_info, status) -> None:
        # Pull whole chunks from the queue until we have enough samples for this block.
        while len(self._buf) < frames:
            try:
                self._buf = np.concatenate([self._buf, self._q.get_nowait()])
            except queue.Empty:
                break
        n = min(frames, len(self._buf))
        outdata[:n] = self._buf[:n]
        if n < frames:
            outdata[n:] = 0.0            # underrun -> brief silence, never a crash
        self._buf = self._buf[n:]

    def write(self, chunk: AudioChunk) -> None:
        if self._stream is None:
            self.open()
        self._q.put(np.ascontiguousarray(chunk.pcm, dtype=np.float32))  # blocks if full

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class NullSink:
    """Headless sink: paces by sleeping for each chunk's duration. No device."""

    def __init__(self, sample_rate: int = 48_000, channels: int = 2) -> None:
        self.sample_rate = sample_rate
        self.channels = channels

    def open(self) -> None:
        pass

    def write(self, chunk: AudioChunk) -> None:
        time.sleep(len(chunk.pcm) / self.sample_rate)

    def close(self) -> None:
        pass
