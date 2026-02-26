"""
Clock and Sequence classes for timing-based operations.
"""

import threading
import time
import warnings
from typing import Optional, Callable, List

from .config import AudioConfig
from .synth import Note, PolySynth


class Clock:
    """Simple clock for timing-based operations."""

    def __init__(self):
        """Initialize clock."""
        self.ticks_per_beat = 4
        self.bpm = 120
        self.tick_length = 0.0
        self.tick_ctr = 0
        self.next_tick = 0.0
        self.start_time_millis = 0
        self.playing = False
        self._shutdown_event = threading.Event()
        self.play_thread: Optional[threading.Thread] = None

        # Callback
        self.on_tick_fns = []

        # Initialize timing
        self.set_bpm(120)

    def on_tick(self):
        for fn in self.on_tick_fns:
            fn()

    def play(self) -> None:
        """Start the clock."""
        if self.playing:
            return

        self.tick_ctr = 0
        self.start_time_millis = int(round(time.time() * 1000))
        self.next_tick = self.tick_length
        self.playing = True
        self._shutdown_event.clear()

        self.play_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self.play_thread.start()

    def stop(self) -> None:
        """Stop the clock."""
        if not self.playing:
            return

        self.playing = False
        self._shutdown_event.set()

        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=AudioConfig.THREAD_JOIN_TIMEOUT)

    def _tick_loop(self) -> None:
        """Main tick loop running in separate thread."""
        while self.playing and not self._shutdown_event.is_set():
            try:
                millis = int(round(time.time() * 1000)) - self.start_time_millis

                if millis >= self.next_tick:
                    self.tick_ctr += 1
                    self.next_tick = millis + self.tick_length

                    try:
                        self.on_tick()
                    except Exception as e:
                        warnings.warn(f"Clock callback error: {e}")

                time.sleep(0.001)
            except Exception as e:
                warnings.warn(f"Clock tick error: {e}")

    def set_bpm(self, bpm: float = 120.0) -> None:
        """
        Set beats per minute.

        Args:
            bpm: Beats per minute
        """
        if bpm <= 0:
            raise ValueError("BPM must be positive")

        self.bpm = bpm
        self.tick_length = 60000.0 / (self.bpm * self.ticks_per_beat)

    def set_tpb(self, ticks_per_beat: int = 4) -> None:
        """
        Set ticks per beat.

        Args:
            ticks_per_beat: Number of ticks per beat
        """
        if ticks_per_beat <= 0:
            raise ValueError("Ticks per beat must be positive")

        self.ticks_per_beat = ticks_per_beat
        self.tick_length = 60000.0 / (self.bpm * self.ticks_per_beat)


class Sequence:
    """Polyphonic step sequencer for driving a PolySynth via a Clock.

    Each step holds a list of :class:`Note` objects.  On every Clock tick
    the sequencer advances and fires ``note_on`` / ``note_off`` events on
    the connected synth.

    *ticks_per_step* controls how many Clock ticks constitute one step, so
    you can set rhythmic resolution independently of the Clock's BPM:

    * ``ticks_per_step=1``  -> one step per tick  (typically 16th notes when
      the Clock's ``ticks_per_beat`` is 4)
    * ``ticks_per_step=4``  -> one step per beat

    Example::

        clock = audio.get_clock(bpm=120)
        clock.set_tpb(4)                    # 4 ticks per beat (16th-note grid)

        synth_idx = audio.start_poly_synth_stream()
        synth = audio.audio_outputs[synth_idx]

        seq = Sequence(steps=8, ticks_per_step=4)   # 8 quarter-note steps
        seq[0] = [Note(60), Note(64)]               # C + E chord
        seq[2] = Note(67)                           # G (single Note OK)
        seq[4] = [Note(60, vel=1.0, duration=2)]    # C, held for 2 steps

        seq.connect(clock, synth)
        clock.play()
    """

    def __init__(self, steps: int = 16, ticks_per_step: int = 1):
        """
        Args:
            steps:          Number of steps in the loop.
            ticks_per_step: Clock ticks per step.
        """
        self._steps: int = steps
        self.ticks_per_step = ticks_per_step
        self._pattern: List[List[Note]] = [[] for _ in range(steps)]
        self._tick_ctr: int = 0
        self._current_step: int = 0
        self._pending_offs: List = []   # list of (tick_off: int, freq: float)
        self._synth: Optional[PolySynth] = None

    # ------------------------------------------------------------------
    # steps property - keeps _current_step in bounds on resize
    # ------------------------------------------------------------------

    @property
    def steps(self) -> int:
        return self._steps

    @steps.setter
    def steps(self, n: int) -> None:
        if n <= 0:
            raise ValueError("steps must be positive")
        if n > self._steps:
            # Grow: pad with empty steps
            self._pattern.extend([] for _ in range(n - self._steps))
        else:
            # Shrink: trim and wrap _current_step into the new range
            self._pattern = self._pattern[:n]
            self._current_step = self._current_step % n
        self._steps = n

    # ------------------------------------------------------------------
    # Pattern editing
    # ------------------------------------------------------------------

    def __setitem__(self, step: int, notes) -> None:
        """Assign notes to a step.  Accepts a single Note or a list."""
        if isinstance(notes, Note):
            notes = [notes]
        self._pattern[step % self._steps] = list(notes)

    def __getitem__(self, step: int) -> List[Note]:
        return self._pattern[step % self.steps]

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def all_notes_off(self) -> None:
        """Immediately send note_off for every note currently pending release."""
        if self._synth is not None:
            for _off_tick, freq in self._pending_offs:
                self._synth.note_off(freq)
        self._pending_offs = []

    def connect(self, clock: 'Clock', synth: 'PolySynth') -> None:
        """Attach this sequence to *clock* and *synth*, replacing the
        clock's ``on_tick`` callback.  Call before ``clock.play()``."""
        self.all_notes_off()
        self._synth = synth
        self._tick_ctr = 0
        self._current_step = 0
        self._pending_offs = []
        clock.on_tick_fns.append(self._on_tick)

    # ------------------------------------------------------------------
    # Tick handler
    # ------------------------------------------------------------------

    def _on_tick(self) -> None:
        # --- Release notes whose duration has elapsed --------------------
        still_pending = []
        for off_tick, freq in self._pending_offs:
            if self._tick_ctr >= off_tick:
                if self._synth is not None:
                    self._synth.note_off(freq)
            else:
                still_pending.append((off_tick, freq))
        self._pending_offs = still_pending

        # --- Advance to next step if on a step boundary ------------------
        if self._tick_ctr % self.ticks_per_step == 0:
            self._current_step = (self._tick_ctr // self.ticks_per_step) % self.steps
            for note in self._pattern[self._current_step]:
                if self._synth is not None:
                    # Cancel any stale pending-off for this freq so it won't
                    # kill the new voice after it is triggered.
                    self._pending_offs = [
                        (t, f) for t, f in self._pending_offs if f != note.freq
                    ]
                    self._synth.note_on(
                        note.freq, note.vel,
                        attack=note.attack, decay=note.decay,
                        sustain=note.sustain, release=note.release,
                        waveform=note.waveform,
                        fm_ratio=note.fm_ratio, fm_index=note.fm_index,
                        detune=note.detune, n_oscs=note.n_oscs,
                        pwm=note.pwm,
                    )
                    off_tick = self._tick_ctr + note.duration * self.ticks_per_step
                    self._pending_offs.append((off_tick, note.freq))

        self._tick_ctr += 1

    # ------------------------------------------------------------------
    # Live-editing helpers
    # ------------------------------------------------------------------

    def clear(self, step: Optional[int] = None) -> None:
        """Silence one step or the entire pattern.

        Args:
            step: Step index to clear.  If ``None``, all steps are cleared.
        """
        if step is None:
            for i in range(self.steps):
                self._pattern[i] = []
        else:
            self._pattern[step % self.steps] = []

    def set_pattern(self, pattern: List[List['Note']]) -> None:
        """Replace the full pattern atomically.

        The new pattern must have exactly ``self.steps`` entries (one list
        of Notes per step).  The change is applied between ticks so no
        notes are dropped or duplicated.

        Args:
            pattern: List of ``self.steps`` lists of :class:`Note`.

        Example::

            seq.set_pattern([
                [Note(60)],   # step 0 - C4
                [],           # step 1 - rest
                [Note(64)],   # step 2 - E4
                [],           # step 3 - rest
            ])
        """
        if len(pattern) != self.steps:
            raise ValueError(
                f"Pattern length {len(pattern)} doesn't match steps {self.steps}"
            )
        self.all_notes_off()
        # Replace each step in-place so _current_step indices remain valid
        for i, notes in enumerate(pattern):
            self._pattern[i] = list(notes)
