"""
Sampler: an AudioDevice that plays audio samples via the Sequence/Note API.
"""

import math
import queue
import warnings
from typing import List

import numpy as np
import numpy.typing as npt
import librosa

from .device import AudioDevice


class Sampler(AudioDevice):
    """Polyphonic sample player driven by the Sequence/Note API.

    Load samples with :meth:`load`, then connect to a :class:`Sequence` and
    :class:`Clock` exactly like a :class:`PolySynth`.

    Each step's ``Note.midi`` is used as the sample slot index (0 = first
    loaded sample, 1 = second, etc.) and ``Note.vel`` scales the playback
    volume.  Samples play to their natural end; ``note_off`` is a no-op so
    one-shots always complete.  You can also trigger samples directly with
    :meth:`trigger`.

    Example::

        sampler_idx = audio.start_sampler_stream(["kick.wav", "snare.wav"])
        sampler = audio.audio_outputs[sampler_idx]

        seq = Sequence(steps=8, ticks_per_step=4)
        seq[0] = Note(0, vel=1.0)   # slot 0 = kick
        seq[2] = Note(1, vel=0.8)   # slot 1 = snare
        seq.connect(clock, sampler)
        clock.play()
    """

    def __init__(self, sr: int = 44100, buffer_size: int = 512, **kwargs):
        super().__init__(
            sr=sr,
            buffer_size=buffer_size,
            use_streaming_analysis=False,
            **kwargs,
        )
        self.samples: List[npt.NDArray[np.float32]] = []
        # Each active voice: {'data': array, 'pos': int, 'vel': float}
        self._voices: List[dict] = []
        self._note_queue: queue.Queue = queue.Queue()

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------

    def load(self, paths: List[str]) -> None:
        """Load audio samples from file paths, resampling to the device sr.

        Args:
            paths: File paths in slot order.  Slot 0 = paths[0], etc.
        """
        self.samples = []
        for path in paths:
            try:
                y, _ = librosa.load(path, sr=self.sr, mono=True)
                self.samples.append(y.astype(np.float32))
            except Exception as e:
                warnings.warn(f"Could not load sample {path}: {e}")
                self.samples.append(np.zeros(self.buffer_size, dtype=np.float32))

    # ------------------------------------------------------------------
    # Sequence/PolySynth-compatible API
    # ------------------------------------------------------------------

    def note_on(self, freq: float, vel: float = 0.8, **_) -> None:
        """Trigger the sample whose slot matches the MIDI note for *freq*.

        ``Note.midi`` 0 → slot 0, 1 → slot 1, etc.
        Safe to call from any thread.
        """
        slot = round(12.0 * math.log2(freq / 440.0) + 69)
        self._note_queue.put(('on', slot, vel))

    def note_off(self, _: float) -> None:
        """No-op: samples always play to their natural end."""

    def all_notes_off(self) -> None:
        """Stop all currently-playing sample voices immediately."""
        self._note_queue.put(('all_off',))

    def trigger(self, index: int, vel: float = 1.0) -> None:
        """Directly trigger a sample by slot index, bypassing MIDI mapping.

        Args:
            index: Slot index (0-based).
            vel:   Volume scale 0.0–1.0.
        """
        self._note_queue.put(('on', index, vel))

    # ------------------------------------------------------------------
    # Internal helpers (audio thread only)
    # ------------------------------------------------------------------

    def _process_queue(self) -> None:
        while not self._note_queue.empty():
            try:
                event = self._note_queue.get_nowait()
                if event[0] == 'on':
                    _, slot, vel = event
                    if 0 <= slot < len(self.samples):
                        self._voices.append({
                            'data': self.samples[slot],
                            'pos': 0,
                            'vel': vel,
                        })
                elif event[0] == 'all_off':
                    self._voices.clear()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # AudioDevice interface
    # ------------------------------------------------------------------

    def audio_callback(self) -> npt.NDArray[np.float32]:
        self._process_queue()

        output = np.zeros(self.buffer_size, dtype=np.float32)
        next_voices = []

        for voice in self._voices:
            data = voice['data']
            pos = voice['pos']
            vel = voice['vel']
            end = pos + self.buffer_size

            if end >= len(data):
                remaining = len(data) - pos
                if remaining > 0:
                    output[:remaining] += data[pos:pos + remaining] * vel
                # voice has finished — don't carry it forward
            else:
                output += data[pos:end] * vel
                voice['pos'] = end
                next_voices.append(voice)

        self._voices = next_voices

        peak = np.max(np.abs(output))
        if peak > 1.0:
            output /= peak

        self.on_new_frame(output)
        self.internal_callback()
        return output * self.gain
