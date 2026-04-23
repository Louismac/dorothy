"""
Audio effects chain: AudioFX base class and built-in effects.

All effects are instantiated with their parameters, then added to an
AudioDevice via ``device.add_effect(fx)``.  Parameters are plain
attributes — write to them at any time (e.g. in draw()) to modulate live.

    lpf = synth.add_effect(LowPassFilter(800))
    lpf.cutoff = 400                # changes take effect on the next buffer
    synth.add_effect(Reverb(wet=0.4))
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import numpy.typing as npt
from scipy import signal as scipy_signal


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class AudioFX:
    """Base class for all audio effects.

    Subclasses override ``process(signal) -> signal``.  Parameters can be
    read/written as plain attributes at any time.
    """

    enabled: bool = True
    sr: int = 44100
    buffer_size: int = 512

    def _init(self, sr: int, buffer_size: int) -> None:
        """Called by ``AudioDevice.add_effect()``.  Store sr/buffer_size and
        allocate any buffers or compute coefficients that depend on them."""
        self.sr = sr
        self.buffer_size = buffer_size

    def process(self, signal: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Process one buffer of audio.  Must return the same length array."""
        return signal


# ---------------------------------------------------------------------------
# Biquad filters (LPF / HPF / BPF)
# ---------------------------------------------------------------------------

class _BiquadFilter(AudioFX):
    """Resonant 2nd-order IIR filter with persistent state between buffers.

    Coefficients use the Audio EQ Cookbook formulas; state is preserved via
    ``scipy.signal.lfilter`` so there are no clicks at buffer boundaries.
    """

    _ftype: str = 'lowpass'   # override in subclasses

    def __init__(self, cutoff: float, q: float = 0.707):
        """
        Args:
            cutoff: Cutoff / centre frequency in Hz.
            q:      Quality factor / resonance (0.1–10, default 0.707 = Butterworth).
        """
        self.cutoff = cutoff
        self.q = q
        self._b: Optional[npt.NDArray] = None
        self._a: Optional[npt.NDArray] = None
        self._z: npt.NDArray = np.zeros(2, dtype=np.float64)
        self._cached_cutoff: float = -1.0
        self._cached_q: float = -1.0

    def _init(self, sr: int, buffer_size: int) -> None:
        super()._init(sr, buffer_size)
        self._z = np.zeros(2, dtype=np.float64)
        self._recompute(force=True)

    def _recompute(self, force: bool = False) -> None:
        if not force and self.cutoff == self._cached_cutoff and self.q == self._cached_q:
            return
        # Clamp to safe range
        f0 = float(np.clip(self.cutoff, 1.0, self.sr * 0.499))
        q = float(np.clip(self.q, 0.01, 100.0))

        w0 = 2.0 * math.pi * f0 / self.sr
        alpha = math.sin(w0) / (2.0 * q)
        cw = math.cos(w0)

        if self._ftype == 'lowpass':
            b0 = (1.0 - cw) / 2.0
            b1 = 1.0 - cw
            b2 = (1.0 - cw) / 2.0
        elif self._ftype == 'highpass':
            b0 = (1.0 + cw) / 2.0
            b1 = -(1.0 + cw)
            b2 = (1.0 + cw) / 2.0
        else:  # bandpass
            b0 = alpha
            b1 = 0.0
            b2 = -alpha

        a0 = 1.0 + alpha
        a1 = -2.0 * cw
        a2 = 1.0 - alpha

        self._b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        self._a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)

        # Reset state to zero on coefficient change (avoids instability from
        # incompatible state, at the cost of a very brief transient).
        self._z = np.zeros(2, dtype=np.float64)
        self._cached_cutoff = self.cutoff
        self._cached_q = self.q

    def process(self, signal: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if not self.enabled:
            return signal
        self._recompute()
        out, self._z = scipy_signal.lfilter(
            self._b, self._a, signal.astype(np.float64), zi=self._z,
        )
        return out.astype(np.float32)


class LowPassFilter(_BiquadFilter):
    """Resonant low-pass filter.

    Args:
        cutoff: Cutoff frequency in Hz.
        q:      Resonance (default 0.707 = Butterworth, >1 adds a resonant peak).
    """
    _ftype = 'lowpass'


class HighPassFilter(_BiquadFilter):
    """Resonant high-pass filter.

    Args:
        cutoff: Cutoff frequency in Hz.
        q:      Resonance (default 0.707 = Butterworth).
    """
    _ftype = 'highpass'


class BandPassFilter(_BiquadFilter):
    """Band-pass filter (constant 0 dB peak gain).

    Args:
        cutoff: Centre frequency in Hz.
        q:      Quality factor (higher = narrower band).
    """
    _ftype = 'bandpass'


# ---------------------------------------------------------------------------
# Gain
# ---------------------------------------------------------------------------

class Gain(AudioFX):
    """Simple dB gain stage.

    Args:
        db: Gain in decibels (negative = attenuate, positive = boost).
    """

    def __init__(self, db: float = 0.0):
        self.db = db

    def process(self, signal: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if not self.enabled:
            return signal
        return (signal * (10.0 ** (self.db / 20.0))).astype(np.float32)


# ---------------------------------------------------------------------------
# Distortion
# ---------------------------------------------------------------------------

class Distortion(AudioFX):
    """Soft-clip saturation via tanh.

    Args:
        drive: Input gain before clipping (1 = clean, higher = more saturation).
        mix:   Wet/dry mix 0–1.
    """

    def __init__(self, drive: float = 2.0, mix: float = 1.0):
        self.drive = drive
        self.mix = mix

    def process(self, signal: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if not self.enabled:
            return signal
        # tanh clip, then normalise so unity gain at drive=1
        norm = float(np.tanh(self.drive)) or 1e-6
        wet = np.tanh(signal * self.drive) / norm
        return (signal * (1.0 - self.mix) + wet * self.mix).astype(np.float32)


# ---------------------------------------------------------------------------
# Delay
# ---------------------------------------------------------------------------

class Delay(AudioFX):
    """Feedback delay line.

    Args:
        time_ms:  Delay time in milliseconds (up to 2 s).
        feedback: Fraction of output fed back into the delay (0–1).
        wet:      Wet/dry mix 0–1.
    """

    def __init__(self, time_ms: float = 250.0, feedback: float = 0.4, wet: float = 0.3):
        self.time_ms = time_ms
        self.feedback = feedback
        self.wet = wet
        self._buf: Optional[npt.NDArray[np.float32]] = None
        self._ptr: int = 0
        self._buf_len: int = 0

    def _init(self, sr: int, buffer_size: int) -> None:
        super()._init(sr, buffer_size)
        self._buf_len = sr * 2   # supports up to 2 s delay
        self._buf = np.zeros(self._buf_len, dtype=np.float32)
        self._ptr = 0

    def process(self, signal: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if not self.enabled:
            return signal
        delay_samps = int(np.clip(self.time_ms * self.sr / 1000.0, 1, self._buf_len - 1))
        n = len(signal)
        out = np.empty(n, dtype=np.float32)
        buf = self._buf

        for i in range(n):
            r = (self._ptr - delay_samps) % self._buf_len
            d = buf[r]
            out[i] = signal[i] + self.wet * d
            buf[self._ptr] = signal[i] + self.feedback * d
            self._ptr = (self._ptr + 1) % self._buf_len

        return out


# ---------------------------------------------------------------------------
# Reverb — canonical Freeverb (8 parallel comb + 4 series allpass)
# ---------------------------------------------------------------------------

# Canonical Freeverb constants (tuned for 44 100 Hz, scaled for other rates)
_FV_COMB_DELAYS  = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
_FV_AP_DELAYS    = [556, 441, 341, 225]
_FV_FIXED_GAIN   = 0.015   # input pre-scale; prevents saturation in the network
_FV_SCALE_WET    = 3.0     # compensates for fixed_gain attenuation
_FV_SCALE_DAMP   = 0.4     # maps 0-1 damping knob → actual damp1 coefficient
_FV_SCALE_ROOM   = 0.28    # }
_FV_OFFSET_ROOM  = 0.7     # } feedback = 0.7 + room * 0.28  (Freeverb standard)


class _CombFilter:
    """Freeverb comb filter with a stateful one-pole LP in the feedback path."""

    def __init__(self, delay_samps: int, feedback: float, damp1: float):
        self._buf = np.zeros(delay_samps, dtype=np.float32)
        self._delay = delay_samps
        self._ptr = 0
        self._filterstore = 0.0   # one-pole LP state
        self.feedback = feedback
        self.damp1 = float(damp1)
        self.damp2 = 1.0 - self.damp1

    def set_damping(self, damp1: float) -> None:
        self.damp1 = float(damp1)
        self.damp2 = 1.0 - self.damp1

    def process(self, signal: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        n = len(signal)
        delay = self._delay
        buf = self._buf
        ptr = self._ptr

        # Read n delayed samples (Freeverb comb delays are all > 512 so no wrap overlap)
        if ptr + n <= delay:
            output = buf[ptr:ptr + n].copy()
        else:
            split = delay - ptr
            output = np.concatenate([buf[ptr:], buf[:n - split]])

        # Stateful one-pole LP applied to delayed signal for the feedback path:
        # filterstore[i] = output[i] * damp2 + filterstore[i-1] * damp1
        filtered_f64, new_zi = scipy_signal.lfilter(
            [self.damp2], [1.0, -self.damp1],
            output.astype(np.float64),
            zi=np.array([self._filterstore]),
        )
        self._filterstore = float(new_zi[0])

        # Write back: input + LP-filtered-delayed * feedback
        write = (signal + filtered_f64.astype(np.float32) * self.feedback)
        if ptr + n <= delay:
            buf[ptr:ptr + n] = write
        else:
            split = delay - ptr
            buf[ptr:] = write[:split]
            buf[:n - split] = write[split:]

        self._ptr = (ptr + n) % delay
        return output


class _AllpassFilter:
    """Freeverb allpass diffuser.

    Processes in chunks of at most ``delay`` samples so that short-delay
    allpasses (delay < buffer_size) correctly feed their own output back
    within the same callback — identical to the sample-by-sample original.
    """

    def __init__(self, delay_samps: int, gain: float = 0.5):
        self._buf = np.zeros(delay_samps, dtype=np.float32)
        self._delay = delay_samps
        self.gain = gain
        self._ptr = 0

    def process(self, signal: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        n = len(signal)
        delay = self._delay
        buf = self._buf
        output = np.empty(n, dtype=np.float32)
        pos = 0

        while pos < n:
            ptr = self._ptr
            # Stay within the current linear run of the ring buffer so that
            # numpy slices are contiguous and don't overlap with the write.
            chunk = min(n - pos, delay - ptr)

            bufout = buf[ptr:ptr + chunk].copy()
            output[pos:pos + chunk] = -signal[pos:pos + chunk] + bufout
            buf[ptr:ptr + chunk] = signal[pos:pos + chunk] + bufout * self.gain

            self._ptr = (ptr + chunk) % delay
            pos += chunk

        return output


class Reverb(AudioFX):
    """Canonical Freeverb algorithmic reverb: 8 parallel comb + 4 series allpass.

    Args:
        room_size: Tail length 0–1 (maps to comb feedback 0.7–0.98).
        damping:   High-frequency absorption 0–1 (higher = darker, shorter tail).
        wet:       Wet/dry mix 0–1.
    """

    def __init__(self, room_size: float = 0.5, damping: float = 0.5, wet: float = 0.33):
        self.room_size = room_size
        self.damping = damping
        self.wet = wet
        self._combs: List[_CombFilter] = []
        self._allpasses: List[_AllpassFilter] = []
        self._last_room = -1.0
        self._last_damp = -1.0

    def _feedback(self) -> float:
        return _FV_OFFSET_ROOM + float(np.clip(self.room_size, 0.0, 1.0)) * _FV_SCALE_ROOM

    def _damp1(self) -> float:
        return float(np.clip(self.damping, 0.0, 1.0)) * _FV_SCALE_DAMP

    def _init(self, sr: int, buffer_size: int) -> None:
        super()._init(sr, buffer_size)
        scale = sr / 44100.0
        fb = self._feedback()
        d1 = self._damp1()
        self._combs = [
            _CombFilter(max(4, round(d * scale)), fb, d1)
            for d in _FV_COMB_DELAYS
        ]
        self._allpasses = [
            _AllpassFilter(max(4, round(d * scale)))
            for d in _FV_AP_DELAYS
        ]
        self._last_room = self.room_size
        self._last_damp = self.damping

    def _sync_params(self) -> None:
        if self.room_size == self._last_room and self.damping == self._last_damp:
            return
        fb = self._feedback()
        d1 = self._damp1()
        for comb in self._combs:
            comb.feedback = fb
            comb.set_damping(d1)
        self._last_room = self.room_size
        self._last_damp = self.damping

    def process(self, signal: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if not self.enabled:
            return signal
        self._sync_params()

        inp = signal * _FV_FIXED_GAIN

        wet_out = np.zeros(len(signal), dtype=np.float32)
        for comb in self._combs:
            wet_out += comb.process(inp)

        for ap in self._allpasses:
            wet_out = ap.process(wet_out)

        return (signal * (1.0 - self.wet) + wet_out * (_FV_SCALE_WET * self.wet)).astype(np.float32)


# ---------------------------------------------------------------------------
# Chorus
# ---------------------------------------------------------------------------

class Chorus(AudioFX):
    """Chorus / flanger effect via a sinusoidally modulated delay line.

    Args:
        rate:  LFO frequency in Hz (default 1.0).
        depth: Modulation depth 0–1 (maps to ±12.5 ms delay swing).
        wet:   Wet/dry mix 0–1.
    """

    _BASE_DELAY_MS = 15.0   # centre delay in ms
    _MAX_SWING_MS = 12.5    # max modulation swing in ms

    def __init__(self, rate: float = 1.0, depth: float = 0.3, wet: float = 0.5):
        self.rate = rate
        self.depth = depth
        self.wet = wet
        self._buf: Optional[npt.NDArray[np.float32]] = None
        self._buf_len: int = 0
        self._ptr: int = 0
        self._lfo_phase: float = 0.0

    def _init(self, sr: int, buffer_size: int) -> None:
        super()._init(sr, buffer_size)
        max_ms = self._BASE_DELAY_MS + self._MAX_SWING_MS
        self._buf_len = int(sr * max_ms / 1000.0) + buffer_size + 8
        self._buf = np.zeros(self._buf_len, dtype=np.float32)
        self._ptr = 0
        self._lfo_phase = 0.0

    def process(self, signal: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if not self.enabled:
            return signal
        n = len(signal)
        buf = self._buf
        buf_len = self._buf_len
        base_delay = int(self._BASE_DELAY_MS * self.sr / 1000.0)
        max_swing = int(self._MAX_SWING_MS * self.sr / 1000.0)
        lfo_inc = 2.0 * math.pi * self.rate / self.sr

        # Precompute per-sample LFO and read positions
        lfo_phases = self._lfo_phase + np.arange(n, dtype=np.float64) * lfo_inc
        mod = (np.sin(lfo_phases) * self.depth * max_swing).astype(np.int32)
        delay_samps = np.clip(base_delay + mod, 1, buf_len - 1)

        write_pos = (self._ptr + np.arange(n)) % buf_len
        read_pos = (write_pos - delay_samps) % buf_len

        # Read OLD buffer state first, then write new samples
        delayed = buf[read_pos]
        buf[write_pos] = signal

        self._ptr = (self._ptr + n) % buf_len
        self._lfo_phase = float(lfo_phases[-1]) + lfo_inc

        return (signal * (1.0 - self.wet) + delayed * self.wet).astype(np.float32)
