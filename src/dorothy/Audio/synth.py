"""
Synthesizer classes: Note, SynthVoice, PolySynth.
"""

import queue
import threading
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from .device import AudioDevice


@dataclass
class Note:
    """A musical note for use in a Sequence.

    Attributes:
        midi:     MIDI note number (0-127, middle C = 60)
        vel:      Velocity 0.0-1.0
        duration: Duration in Sequence steps
        attack:   Override synth attack time (seconds), or None to use synth default
        decay:    Override synth decay time (seconds), or None to use synth default
        sustain:  Override synth sustain level (0-1), or None to use synth default
        release:  Override synth release time (seconds), or None to use synth default
    """
    midi: int
    vel: float = 0.8
    duration: int = 1
    attack: Optional[float] = None
    decay: Optional[float] = None
    sustain: Optional[float] = None
    release: Optional[float] = None
    waveform: Optional[str] = None   # 'sine'|'saw'|'triangle'|'noise'|'supersaw'|'fm'|'pwm'
    # FM params
    fm_ratio: Optional[float] = None  # modulator freq = fm_ratio * carrier freq
    fm_index: Optional[float] = None  # modulation depth in radians
    # Supersaw params
    detune: Optional[float] = None    # total semitone spread across all oscillators
    n_oscs: Optional[int] = None      # number of detuned oscillators
    # PWM param
    pwm: Optional[float] = None       # duty cycle 0-1 (0.5 = square)

    @property
    def freq(self) -> float:
        """Convert MIDI note number to frequency in Hz."""
        return 440.0 * (2.0 ** ((self.midi - 69) / 12.0))


class SynthVoice:
    """Single additive synthesis voice with ADSR envelope.

    Generates audio as a sum of harmonically-related sinusoids,
    shaped by an ADSR amplitude envelope.  All rendering is done
    with vectorised numpy so it is safe to call from the audio
    callback thread.
    """

    def __init__(self, sr: int, n_harmonics: int = 4):
        self.sr = sr
        self.n_harmonics = n_harmonics
        self.freq = 440.0
        self.vel = 0.8
        self.active = False

        # Phase accumulator per harmonic (radians)
        self.phases = np.zeros(n_harmonics, dtype=np.float64)

        # Harmonic amplitudes for 'sine' mode: 1/h roll-off, normalised
        raw = np.array([1.0 / (h + 1) for h in range(n_harmonics)], dtype=np.float32)
        self.harmonic_amps = raw / raw.sum()

        # Oscillator waveform: 'sine'|'saw'|'triangle'|'noise'|'supersaw'|'fm'|'pwm'
        self.waveform: str = 'sine'

        # FM params
        self.fm_ratio: float = 2.0    # modulator = fm_ratio * carrier
        self.fm_index: float = 1.0    # modulation depth (radians)
        self.mod_phase: float = 0.0   # modulator phase accumulator

        # Supersaw params
        self.n_oscs: int = 7          # number of detuned saws
        self.detune: float = 0.2      # total semitone spread across all oscs
        self.osc_phases: npt.NDArray[np.float64] = np.zeros(7, dtype=np.float64)

        # PWM param
        self.pwm: float = 0.5         # duty cycle (0.5 = square wave)

        # ADSR parameters (seconds / level)
        self.attack: float = 0.01
        self.decay: float = 0.1
        self.sustain: float = 0.7
        self.release: float = 0.3

        # Envelope state machine
        self.env_state: str = 'idle'   # idle | attack | decay | sustain | release
        self.env_sample: int = 0       # samples elapsed in current state
        self.env_level: float = 0.0    # current envelope amplitude
        self.release_start_level: float = 0.0

    def note_on(self, freq: float, vel: float) -> None:
        self.freq = freq
        self.vel = vel
        self.phases[:] = 0.0
        self.mod_phase = 0.0
        self.osc_phases = np.zeros(self.n_oscs, dtype=np.float64)
        self.env_state = 'attack'
        self.env_sample = 0
        self.env_level = 0.0
        self.active = True

    def note_off(self) -> None:
        if self.active and self.env_state != 'release':
            self.release_start_level = self.env_level
            self.env_state = 'release'
            self.env_sample = 0

    def _generate_envelope(self, n_samples: int) -> npt.NDArray[np.float32]:
        """Generate n_samples of envelope, advancing the state machine."""
        env = np.zeros(n_samples, dtype=np.float32)
        pos = 0

        while pos < n_samples:
            if self.env_state == 'idle':
                break

            elif self.env_state == 'attack':
                atk = max(int(self.attack * self.sr), 1)
                remaining = atk - self.env_sample
                count = min(remaining, n_samples - pos)
                if count > 0:
                    t = np.arange(self.env_sample, self.env_sample + count,
                                  dtype=np.float32) / atk
                    env[pos:pos + count] = t
                    self.env_level = float(t[-1])
                self.env_sample += count
                pos += count
                if self.env_sample >= atk:
                    self.env_state = 'decay'
                    self.env_sample = 0
                    self.env_level = 1.0

            elif self.env_state == 'decay':
                dec = max(int(self.decay * self.sr), 1)
                remaining = dec - self.env_sample
                count = min(remaining, n_samples - pos)
                if count > 0:
                    t = np.arange(self.env_sample, self.env_sample + count,
                                  dtype=np.float32) / dec
                    vals = 1.0 - t * (1.0 - self.sustain)
                    env[pos:pos + count] = vals
                    self.env_level = float(vals[-1])
                self.env_sample += count
                pos += count
                if self.env_sample >= dec:
                    self.env_state = 'sustain'
                    self.env_sample = 0
                    self.env_level = self.sustain

            elif self.env_state == 'sustain':
                count = n_samples - pos
                env[pos:pos + count] = self.sustain
                self.env_level = self.sustain
                pos += count

            elif self.env_state == 'release':
                rel = max(int(self.release * self.sr), 1)
                remaining = rel - self.env_sample
                count = min(remaining, n_samples - pos)
                if count > 0:
                    t = np.arange(self.env_sample, self.env_sample + count,
                                  dtype=np.float32) / rel
                    vals = self.release_start_level * (1.0 - t)
                    env[pos:pos + count] = vals
                    self.env_level = float(vals[-1])
                self.env_sample += count
                pos += count
                if self.env_sample >= rel:
                    self.env_state = 'idle'
                    self.active = False
                    self.env_level = 0.0
                    break

        return env

    def _render_osc(self, n_samples: int) -> npt.NDArray[np.float32]:
        """Generate raw oscillator samples (no envelope, no velocity)."""
        idx = np.arange(n_samples, dtype=np.float64)
        phase_inc = 2.0 * np.pi * self.freq / self.sr

        if self.waveform == 'sine':
            # Additive stack: fundamental + harmonics with 1/h roll-off
            output = np.zeros(n_samples, dtype=np.float32)
            for h in range(self.n_harmonics):
                harmonic_freq = self.freq * (h + 1)
                if harmonic_freq >= self.sr * 0.5:
                    break
                h_inc = phase_inc * (h + 1)
                output += (self.harmonic_amps[h] *
                           np.sin(self.phases[h] + idx * h_inc)).astype(np.float32)
                self.phases[h] = (self.phases[h] + n_samples * h_inc) % (2.0 * np.pi)

        elif self.waveform == 'saw':
            # Rising sawtooth: -1 to +1 per period
            phases = self.phases[0] + idx * phase_inc
            output = ((phases / np.pi) % 2.0 - 1.0).astype(np.float32)
            self.phases[0] = (self.phases[0] + n_samples * phase_inc) % (2.0 * np.pi)

        elif self.waveform == 'triangle':
            # Triangle: -1 to +1 to -1 per period
            phases = self.phases[0] + idx * phase_inc
            t = (phases / (2.0 * np.pi)) % 1.0
            output = (1.0 - 4.0 * np.abs(t - 0.5)).astype(np.float32)
            self.phases[0] = (self.phases[0] + n_samples * phase_inc) % (2.0 * np.pi)

        elif self.waveform == 'supersaw':
            # N detuned sawtooth oscillators - spread evenly across detune semitones
            output = np.zeros(n_samples, dtype=np.float32)
            n = self.n_oscs
            spread_oct = self.detune / 12.0   # semitones to octave fraction
            for k in range(n):
                # evenly distribute from -spread/2 to +spread/2
                t = (k / (n - 1) - 0.5) if n > 1 else 0.0
                f = self.freq * (2.0 ** (spread_oct * t))
                if f >= self.sr * 0.5:
                    continue
                f_inc = 2.0 * np.pi * f / self.sr
                phases = self.osc_phases[k] + idx * f_inc
                output += ((phases / np.pi) % 2.0 - 1.0).astype(np.float32)
                self.osc_phases[k] = (self.osc_phases[k] + n_samples * f_inc) % (2.0 * np.pi)
            output /= n

        elif self.waveform == 'fm':
            # 2-operator FM: output = sin(carrier + fm_index * sin(modulator))
            mod_freq = self.freq * self.fm_ratio
            if mod_freq < self.sr * 0.5:
                mod_inc = 2.0 * np.pi * mod_freq / self.sr
                mod_signal = self.fm_index * np.sin(self.mod_phase + idx * mod_inc)
                self.mod_phase = (self.mod_phase + n_samples * mod_inc) % (2.0 * np.pi)
            else:
                mod_signal = 0.0
            carrier_phases = self.phases[0] + idx * phase_inc
            output = np.sin(carrier_phases + mod_signal).astype(np.float32)
            self.phases[0] = (self.phases[0] + n_samples * phase_inc) % (2.0 * np.pi)

        elif self.waveform == 'pwm':
            # Pulse-width modulation: duty cycle sets ratio of high to low
            phases = self.phases[0] + idx * phase_inc
            norm = (phases % (2.0 * np.pi)) / (2.0 * np.pi)
            output = np.where(norm < self.pwm, 1.0, -1.0).astype(np.float32)
            self.phases[0] = (self.phases[0] + n_samples * phase_inc) % (2.0 * np.pi)

        else:
            output = np.zeros(n_samples, dtype=np.float32)

        return output

    def render(self, n_samples: int) -> npt.NDArray[np.float32]:
        """Render n_samples of audio."""
        if not self.active:
            return np.zeros(n_samples, dtype=np.float32)

        env = self._generate_envelope(n_samples)

        if self.waveform == 'noise':
            output = np.random.uniform(-1.0, 1.0, n_samples).astype(np.float32)
        else:
            output = self._render_osc(n_samples)

        return output * env * self.vel


class PolySynth(AudioDevice):
    """Polyphonic additive synthesizer AudioDevice.

    Renders up to *n_voices* simultaneous notes.  Each voice uses additive
    synthesis (a stack of harmonics) shaped by an ADSR envelope.  Note
    events are posted via a thread-safe queue so they can be scheduled from
    Clock callbacks or user code without blocking the audio thread.

    ADSR parameters can be adjusted at any time via the corresponding
    properties (attack, decay, sustain, release); changes apply to all voices.

    Harmonic amplitudes follow a 1/h roll-off (h = harmonic index 1...n),
    normalised so the total power is independent of *n_harmonics*.

    Example::

        synth_idx = audio.start_poly_synth_stream(n_voices=6, n_harmonics=6)
        synth = audio.audio_outputs[synth_idx]
        synth.note_on(440.0, vel=0.8)   # A4
        synth.note_off(440.0)
    """

    WAVEFORMS = {'sine', 'saw', 'triangle', 'noise', 'supersaw', 'fm', 'pwm'}

    def __init__(
        self,
        n_voices: int = 8,
        n_harmonics: int = 4,
        attack: float = 0.01,
        decay: float = 0.1,
        sustain: float = 0.7,
        release: float = 0.3,
        waveform: str = 'sine',
        fm_ratio: float = 2.0,
        fm_index: float = 1.0,
        detune: float = 0.2,
        n_oscs: int = 7,
        pwm: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_voices = n_voices
        self.n_harmonics = n_harmonics

        self.voices = [SynthVoice(sr=self.sr, n_harmonics=n_harmonics)
                       for _ in range(n_voices)]

        # Defaults applied to every voice at note_on time unless
        # the individual Note provides its own override values.
        self._default_attack: float = attack
        self._default_decay: float = decay
        self._default_sustain: float = sustain
        self._default_release: float = release
        self._default_waveform: str = waveform
        self._default_fm_ratio: float = fm_ratio
        self._default_fm_index: float = fm_index
        self._default_detune: float = detune
        self._default_n_oscs: int = n_oscs
        self._default_pwm: float = pwm

        self._note_queue: queue.Queue = queue.Queue()
        # Maps freq (Hz) -> voice index for note-off matching
        self._active_notes: dict = {}

    # ------------------------------------------------------------------
    # ADSR defaults - readable/writable on the synth object
    # ------------------------------------------------------------------

    @property
    def attack(self) -> float:
        return self._default_attack

    @attack.setter
    def attack(self, val: float) -> None:
        self._default_attack = val

    @property
    def decay(self) -> float:
        return self._default_decay

    @decay.setter
    def decay(self, val: float) -> None:
        self._default_decay = val

    @property
    def sustain(self) -> float:
        return self._default_sustain

    @sustain.setter
    def sustain(self, val: float) -> None:
        self._default_sustain = val

    @property
    def release(self) -> float:
        return self._default_release

    @release.setter
    def release(self, val: float) -> None:
        self._default_release = val

    @property
    def waveform(self) -> str:
        return self._default_waveform

    @waveform.setter
    def waveform(self, val: str) -> None:
        if val not in self.WAVEFORMS:
            raise ValueError(f"waveform must be one of {self.WAVEFORMS}")
        self._default_waveform = val

    @property
    def fm_ratio(self) -> float:
        return self._default_fm_ratio

    @fm_ratio.setter
    def fm_ratio(self, val: float) -> None:
        self._default_fm_ratio = val

    @property
    def fm_index(self) -> float:
        return self._default_fm_index

    @fm_index.setter
    def fm_index(self, val: float) -> None:
        self._default_fm_index = val

    @property
    def detune(self) -> float:
        return self._default_detune

    @detune.setter
    def detune(self, val: float) -> None:
        self._default_detune = val

    @property
    def n_oscs(self) -> int:
        return self._default_n_oscs

    @n_oscs.setter
    def n_oscs(self, val: int) -> None:
        self._default_n_oscs = val

    @property
    def pwm(self) -> float:
        return self._default_pwm

    @pwm.setter
    def pwm(self, val: float) -> None:
        self._default_pwm = val

    # ------------------------------------------------------------------
    # Public note API (thread-safe via queue)
    # ------------------------------------------------------------------

    def note_on(
        self,
        freq: float,
        vel: float = 0.8,
        attack: Optional[float] = None,
        decay: Optional[float] = None,
        sustain: Optional[float] = None,
        release: Optional[float] = None,
        waveform: Optional[str] = None,
        fm_ratio: Optional[float] = None,
        fm_index: Optional[float] = None,
        detune: Optional[float] = None,
        n_oscs: Optional[int] = None,
        pwm: Optional[float] = None,
    ) -> None:
        """Schedule a note-on event.  Safe to call from any thread.

        Per-note values override the synth defaults for this note only.
        Pass ``None`` (the default) to use the synth's current default.
        """
        self._note_queue.put((
            'on', freq, vel,
            attack, decay, sustain, release,
            waveform, fm_ratio, fm_index, detune, n_oscs, pwm,
        ))

    def note_off(self, freq: float) -> None:
        """Schedule a note-off event.  Safe to call from any thread."""
        self._note_queue.put(('off', freq))

    # ------------------------------------------------------------------
    # Internal helpers (called from audio callback thread)
    # ------------------------------------------------------------------

    def _process_note_queue(self) -> None:
        while not self._note_queue.empty():
            try:
                event = self._note_queue.get_nowait()
                if event[0] == 'on':
                    _, freq, vel, atk, dec, sus, rel, wav, fmr, fmi, det, nos, pw = event
                    self._voice_note_on(freq, vel, atk, dec, sus, rel, wav, fmr, fmi, det, nos, pw)
                elif event[0] == 'off':
                    self._voice_note_off(event[1])
            except queue.Empty:
                break

    def _configure_voice(
        self,
        voice: 'SynthVoice',
        attack: Optional[float],
        decay: Optional[float],
        sustain: Optional[float],
        release: Optional[float],
        waveform: Optional[str],
        fm_ratio: Optional[float],
        fm_index: Optional[float],
        detune: Optional[float],
        n_oscs: Optional[int],
        pwm: Optional[float],
    ) -> None:
        """Apply per-note params to a voice, falling back to synth defaults."""
        voice.attack   = attack   if attack   is not None else self._default_attack
        voice.decay    = decay    if decay    is not None else self._default_decay
        voice.sustain  = sustain  if sustain  is not None else self._default_sustain
        voice.release  = release  if release  is not None else self._default_release
        voice.waveform = waveform if waveform is not None else self._default_waveform
        voice.fm_ratio = fm_ratio if fm_ratio is not None else self._default_fm_ratio
        voice.fm_index = fm_index if fm_index is not None else self._default_fm_index
        voice.detune   = detune   if detune   is not None else self._default_detune
        voice.n_oscs   = n_oscs   if n_oscs   is not None else self._default_n_oscs
        voice.pwm      = pwm      if pwm      is not None else self._default_pwm

    def _voice_note_on(
        self,
        freq: float,
        vel: float,
        attack: Optional[float],
        decay: Optional[float],
        sustain: Optional[float],
        release: Optional[float],
        waveform: Optional[str],
        fm_ratio: Optional[float],
        fm_index: Optional[float],
        detune: Optional[float],
        n_oscs: Optional[int],
        pwm: Optional[float],
    ) -> None:
        args = (attack, decay, sustain, release, waveform, fm_ratio, fm_index, detune, n_oscs, pwm)
        # 1. Prefer idle (silent) voices
        for i, voice in enumerate(self.voices):
            if not voice.active:
                self._configure_voice(voice, *args)
                voice.note_on(freq, vel)
                self._active_notes[freq] = i
                return
        # 2. Steal the first releasing voice
        for i, voice in enumerate(self.voices):
            if voice.env_state == 'release':
                self._remove_note_by_voice(i)
                self._configure_voice(voice, *args)
                voice.note_on(freq, vel)
                self._active_notes[freq] = i
                return
        # 3. Last resort: steal voice 0
        self._remove_note_by_voice(0)
        self._configure_voice(self.voices[0], *args)
        self.voices[0].note_on(freq, vel)
        self._active_notes[freq] = 0

    def _voice_note_off(self, freq: float) -> None:
        if freq in self._active_notes:
            self.voices[self._active_notes[freq]].note_off()
            del self._active_notes[freq]

    def _remove_note_by_voice(self, voice_idx: int) -> None:
        stale = [f for f, i in self._active_notes.items() if i == voice_idx]
        for f in stale:
            del self._active_notes[f]

    # ------------------------------------------------------------------
    # AudioDevice interface
    # ------------------------------------------------------------------

    def audio_callback(self) -> npt.NDArray[np.float32]:
        self._process_note_queue()

        output = np.zeros(self.buffer_size, dtype=np.float32)
        for voice in self.voices:
            if voice.active:
                output += voice.render(self.buffer_size)

        # Soft normalise to prevent clipping
        peak = np.max(np.abs(output))
        if peak > 1.0:
            output /= peak

        self.on_new_frame(output)
        self.internal_callback()
        return output * self.gain
