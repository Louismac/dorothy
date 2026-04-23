"""
GranularSynth: granular synthesis AudioDevice.
"""

import queue
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import librosa

from .device import AudioDevice


class GranularSynth(AudioDevice):
    """Granular synthesizer AudioDevice.

    Loads a single audio file and plays it back as overlapping short grains.
    Connects to a :class:`Sequence` and :class:`Clock` exactly like a
    :class:`PolySynth` via ``note_on`` / ``note_off``.

    ``Note.midi`` 69 (A4, 440 Hz) = original file pitch.  Other values
    shift pitch up/down by the corresponding semitone distance from A4.
    ``Note.vel`` scales the voice volume.

    Parameters
    ----------
    position : float
        Centre read position in the source file, normalised 0-1.
    spread : float
        Random scatter around *position*, as a fraction of the file length.
        Each grain picks a random offset in ``[-spread, +spread]``.
    grain_size : float
        Duration of each grain in milliseconds (default 80 ms).
    density : float
        New grains spawned per second per active voice (default 8).
    attack : float
        Fraction of the grain length used for the linear fade-in (0-1).
    decay : float
        Fraction of the grain length used for the linear fade-out (0-1).
    n_grains : int
        Maximum simultaneous grains across all voices.
    pitch : float
        Global pitch shift in semitones applied to every grain.
    pitch_spread : float
        Per-grain random pitch jitter in semitones (Gaussian std dev).

    Example::

        gran_idx = audio.start_granular_stream("texture.wav", density=12)
        gran = audio.audio_outputs[gran_idx]
        gran.position = 0.4
        gran.spread = 0.05

        gran.note_on(440, vel=0.8)   # starts the grain cloud, never stops
    """

    def __init__(
        self,
        sr: int = 44100,
        buffer_size: int = 512,
        position: float = 0.5,
        spread: float = 0.1,
        grain_size: float = 80.0,
        density: float = 8.0,
        attack: float = 0.3,
        decay: float = 0.3,
        n_grains: int = 32,
        pitch: float = 0.0,
        pitch_spread: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            sr=sr,
            buffer_size=buffer_size,
            use_streaming_analysis=False,
            **kwargs,
        )

        # Source material
        self._source: Optional[npt.NDArray[np.float32]] = None

        # --- Synthesis parameters (read/write at any time) ----------------
        self.position: float = position       # 0–1 normalised file position
        self.spread: float = spread           # 0–1 position scatter
        self.grain_size: float = grain_size   # ms per grain
        self.density: float = density         # grains / second / voice
        self.attack: float = attack           # fraction of grain for fade-in
        self.decay: float = decay             # fraction of grain for fade-out
        self.n_grains: int = n_grains         # max simultaneous grains
        self.pitch: float = pitch             # global semitone shift
        self.pitch_spread: float = pitch_spread  # per-grain semitone jitter

        # --- Runtime state (audio thread only) ----------------------------
        # Each voice: {freq, vel, releasing, last_spawn}
        self._voices: List[Dict[str, Any]] = []
        # Each grain: {source_phase, rate, vel, envelope, phase}
        self._grains: List[Dict[str, Any]] = []
        self._sample_ctr: int = 0
        self._note_queue: queue.Queue = queue.Queue()

    # ------------------------------------------------------------------
    # Source loading
    # ------------------------------------------------------------------

    def load(self, path: str) -> None:
        """Load a source audio file for granular playback.

        The file is resampled to the device sample rate and converted to mono.

        Args:
            path: Path to any audio file format supported by librosa.
        """
        try:
            y, _ = librosa.load(path, sr=self.sr, mono=True)
            self._source = y.astype(np.float32)
        except Exception as e:
            warnings.warn(f"GranularSynth: could not load '{path}': {e}")
            self._source = None

    # ------------------------------------------------------------------
    # Sequence / PolySynth-compatible API (thread-safe)
    # ------------------------------------------------------------------

    def note_on(self, freq: float, vel: float = 0.8, **_) -> None:
        """Start a grain cloud at *freq* (pitch) and *vel* (volume).

        440 Hz = original file pitch.  Safe to call from any thread.
        """
        self._note_queue.put(('on', freq, vel))

    def note_off(self, freq: float) -> None:
        """Stop spawning new grains for *freq*.  Active grains play to completion.

        Safe to call from any thread.
        """
        self._note_queue.put(('off', freq))

    def all_notes_off(self) -> None:
        """Immediately silence all voices and clear every active grain."""
        self._note_queue.put(('all_off',))

    # ------------------------------------------------------------------
    # Internal machinery (audio thread)
    # ------------------------------------------------------------------

    def _process_queue(self) -> None:
        while not self._note_queue.empty():
            try:
                event = self._note_queue.get_nowait()
                if event[0] == 'on':
                    _, freq, vel = event
                    # Replace any voice already at this freq
                    self._voices = [v for v in self._voices if v['freq'] != freq]
                    self._voices.append({
                        'freq': freq,
                        'vel': vel,
                        'releasing': False,
                        # Set last_spawn behind by one interval so the first
                        # grain spawns on this buffer rather than one interval later.
                        'last_spawn': self._sample_ctr - int(self.sr / max(self.density, 0.01)),
                    })
                elif event[0] == 'off':
                    for v in self._voices:
                        if v['freq'] == event[1]:
                            v['releasing'] = True
                elif event[0] == 'all_off':
                    self._voices.clear()
                    self._grains.clear()
            except queue.Empty:
                break

    def _build_envelope(self, grain_samps: int) -> npt.NDArray[np.float32]:
        """Trapezoid envelope: linear attack, sustain, linear decay."""
        atk = max(int(self.attack * grain_samps), 1)
        dec = max(int(self.decay * grain_samps), 1)
        sus = max(grain_samps - atk - dec, 0)
        return np.concatenate([
            np.linspace(0.0, 1.0, atk, dtype=np.float32),
            np.ones(sus, dtype=np.float32),
            np.linspace(1.0, 0.0, dec, dtype=np.float32),
        ])

    def _spawn_grain(self, voice: Dict) -> None:
        """Append one new grain driven by *voice* to self._grains."""
        source = self._source
        if source is None or len(source) < 2:
            return
        if len(self._grains) >= self.n_grains:
            return

        source_len = len(source)
        grain_samps = max(int(self.grain_size * self.sr / 1000.0), 2)

        # --- Read position: centre + random scatter ----------------------
        spread_samps = int(self.spread * source_len)
        centre = int(self.position * (source_len - 1))
        offset = int(np.random.uniform(-spread_samps, spread_samps)) if spread_samps > 0 else 0
        src_pos = float(np.clip(centre + offset, 0, source_len - grain_samps - 1))

        # --- Playback rate: global pitch + note pitch + random jitter ----
        semitones = self.pitch
        if self.pitch_spread > 0.0:
            semitones += float(np.random.normal(0.0, self.pitch_spread))
        rate = (2.0 ** (semitones / 12.0)) * (voice['freq'] / 440.0)
        rate = max(rate, 0.01)  # guard against zero / negative rates

        self._grains.append({
            'source_phase': src_pos,
            'rate': rate,
            'vel': voice['vel'],
            'envelope': self._build_envelope(grain_samps),
            'phase': 0,
        })

    def _render_grains(self, output: npt.NDArray[np.float32]) -> None:
        """Mix every active grain into *output* (in-place), remove finished grains."""
        source = self._source
        if source is None:
            return

        source_len = len(source)
        next_grains = []

        for grain in self._grains:
            phase = grain['phase']
            env = grain['envelope']
            grain_len = len(env)
            to_render = min(grain_len - phase, self.buffer_size)

            if to_render <= 0:
                continue

            # Fractional source read positions for this chunk
            src_idx = grain['source_phase'] + np.arange(to_render, dtype=np.float64) * grain['rate']

            # How many indices fall within the source bounds?
            out_of_bounds = src_idx >= (source_len - 1)
            if out_of_bounds.any():
                n_valid = int(np.argmax(out_of_bounds))
            else:
                n_valid = to_render

            if n_valid > 0:
                sv = src_idx[:n_valid]
                lo = sv.astype(np.int32)
                frac = (sv - lo).astype(np.float32)
                hi = np.minimum(lo + 1, source_len - 1)
                samples = source[lo] * (1.0 - frac) + source[hi] * frac
                output[:n_valid] += samples * env[phase:phase + n_valid] * grain['vel']
                grain['source_phase'] = float(src_idx[n_valid - 1]) + grain['rate']

            grain['phase'] = phase + n_valid

            # Keep grain only if envelope is not exhausted and source is not exhausted
            if grain['phase'] < grain_len and n_valid == to_render:
                next_grains.append(grain)

        self._grains = next_grains

    # ------------------------------------------------------------------
    # AudioDevice interface
    # ------------------------------------------------------------------

    def audio_callback(self) -> npt.NDArray[np.float32]:
        self._process_queue()

        output = np.zeros(self.buffer_size, dtype=np.float32)

        if self._source is not None:
            # Spawn grains for active (non-releasing) voices.
            # Use a while loop so high-density settings can spawn multiple
            # grains per buffer without falling behind.
            spawn_interval = max(int(self.sr / max(self.density, 0.01)), 1)
            for voice in self._voices:
                if not voice['releasing']:
                    while self._sample_ctr - voice['last_spawn'] >= spawn_interval:
                        self._spawn_grain(voice)
                        voice['last_spawn'] += spawn_interval

            # Releasing voices spawn no more grains; remove them now.
            self._voices = [v for v in self._voices if not v['releasing']]

            self._render_grains(output)

        self._sample_ctr += self.buffer_size

        # Soft normalise
        peak = np.max(np.abs(output))
        if peak > 1.0:
            output /= peak

        self.on_new_frame(output)
        self.internal_callback()
        return output * self.gain
