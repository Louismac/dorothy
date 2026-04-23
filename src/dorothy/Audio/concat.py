"""
ConcatSynth: descriptor-driven concatenative synthesis AudioDevice.

Subclass of GranularSynth that replaces position+spread grain selection
with a nearest-neighbour lookup over audio descriptors.  All granular
controls (density, grain_size, attack, decay, pitch, n_grains, note_on /
note_off, Sequence/Clock integration) are inherited unchanged.

The ``target`` dict drives which corpus units are selected::

    cat.target['centroid'] = 3000   # prefer brighter grains
    cat.target['rms']      = 0.6    # prefer louder grains

``spread`` is repurposed as intra-unit scatter (0 = start of unit,
1 = anywhere within the unit).
"""

import os
import queue
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import librosa
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .granular import GranularSynth


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def _mfcc(y: npt.NDArray, sr: int) -> npt.NDArray:
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)

def _centroid(y: npt.NDArray, sr: int) -> npt.NDArray:
    return librosa.feature.spectral_centroid(y=y, sr=sr).mean(axis=1)

def _rms(y: npt.NDArray, sr: int) -> npt.NDArray:
    return librosa.feature.rms(y=y).mean(axis=1)

def _flatness(y: npt.NDArray, sr: int) -> npt.NDArray:
    return librosa.feature.spectral_flatness(y=y).mean(axis=1)

def _zcr(y: npt.NDArray, sr: int) -> npt.NDArray:
    return librosa.feature.zero_crossing_rate(y).mean(axis=1)

def _chroma(y: npt.NDArray, sr: int) -> npt.NDArray:
    return librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)


_EXTRACTORS = {
    'mfcc':     _mfcc,
    'centroid': _centroid,
    'rms':      _rms,
    'flatness': _flatness,
    'zcr':      _zcr,
    'chroma':   _chroma,
}

_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.aif', '.aiff', '.ogg', '.m4a', '.opus'}


def _find_audio_files(directory: str) -> List[str]:
    """Recursively collect audio files under *directory*, sorted by path."""
    found = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in _AUDIO_EXTENSIONS:
                found.append(os.path.join(root, fname))
    return sorted(found)


def _extract(y: npt.NDArray, sr: int, feature_list: Sequence[str]) -> npt.NDArray[np.float32]:
    parts = []
    for name in feature_list:
        fn = _EXTRACTORS.get(name)
        if fn is None:
            warnings.warn(f"ConcatSynth: unknown feature '{name}', skipping")
            continue
        parts.append(fn(y, sr).astype(np.float32))
    return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)


# ---------------------------------------------------------------------------
# ConcatSynth
# ---------------------------------------------------------------------------

class ConcatSynth(GranularSynth):
    """Granular synthesizer with descriptor-driven grain selection.

    Inherits all granular controls from :class:`GranularSynth` — density,
    grain_size, attack/decay envelopes, pitch shifting, polyphonic
    note_on/note_off, and Sequence/Clock integration.  The only behavioural
    difference is how each grain is placed: instead of random scatter around
    ``position``, grains are drawn from the corpus unit whose descriptors
    best match the current ``target``.

    ``spread`` is repurposed: 0 = always start at the unit's first sample;
    1 = scatter uniformly across the full unit.  ``position`` is unused.

    Parameters
    ----------
    unit_size : float
        Duration of each corpus unit in milliseconds.  Corpus is sliced at
        this size on load; ``grain_size`` defaults to the same value.
    n_candidates : int
        KNN candidates to randomly pick from (1 = best match every time).
    features : sequence of str
        Descriptors: ``'mfcc'``, ``'centroid'``, ``'rms'``, ``'flatness'``,
        ``'zcr'``, ``'chroma'``.

    Example::

        cat_idx = audio.start_concat_stream("corpus.wav", density=12)
        cat = audio.audio_outputs[cat_idx]
        cat.target['centroid'] = 3000

        seq = Sequence(steps=4, ticks_per_step=8)
        seq[0] = Note(69, vel=0.8)
        seq.connect(clock, cat)
        clock.play()
    """

    def __init__(
        self,
        sr: int = 44100,
        buffer_size: int = 512,
        unit_size: float = 80.0,
        n_candidates: int = 5,
        features: Tuple[str, ...] = ('mfcc', 'centroid', 'rms'),
        # Granular controls passed through to parent
        grain_size: Optional[float] = None,
        density: float = 8.0,
        attack: float = 0.3,
        decay: float = 0.3,
        n_grains: int = 32,
        pitch: float = 0.0,
        pitch_spread: float = 0.0,
        spread: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            sr=sr,
            buffer_size=buffer_size,
            grain_size=grain_size if grain_size is not None else unit_size,
            density=density,
            attack=attack,
            decay=decay,
            n_grains=n_grains,
            pitch=pitch,
            pitch_spread=pitch_spread,
            spread=spread,
            **kwargs,
        )

        self.unit_size: float = unit_size
        self.n_candidates: int = n_candidates
        self.features: Tuple[str, ...] = tuple(features)

        # Target feature values — set any key to steer grain selection.
        self.target: Dict[str, float] = {}

        # KNN corpus state (populated by load())
        self._unit_offsets: Optional[npt.NDArray[np.int64]] = None  # start sample per unit
        self._feat_matrix: Optional[npt.NDArray[np.float32]] = None
        self._feat_names: List[str] = []
        self._scaler: Optional[StandardScaler] = None
        self._knn: Optional[NearestNeighbors] = None
        self._n_units: int = 0
        self._unit_samps: int = 0

        # Queue for bulk target updates from set_target_from_audio()
        self._target_q: queue.Queue = queue.Queue()

    # ------------------------------------------------------------------
    # Corpus loading
    # ------------------------------------------------------------------

    def load(self, path: str) -> None:
        """Load a corpus from a single audio file or a directory.

        When *path* is a directory every audio file found recursively
        is included (wav, mp3, flac, aif, aiff, ogg, m4a, opus).

        Args:
            path: Audio file path or directory path.
        """
        if os.path.isdir(path):
            paths = _find_audio_files(path)
            if not paths:
                warnings.warn(f"ConcatSynth: no audio files found in '{path}'")
                return
            print(f"ConcatSynth: found {len(paths)} files in '{path}'")
        else:
            paths = [path]
        self._load_files(paths)

    def _load_files(self, paths: List[str]) -> None:
        unit_samps = max(int(self.unit_size * self.sr / 1000.0), self.buffer_size)

        all_units: List[npt.NDArray[np.float32]] = []
        all_feats: List[npt.NDArray[np.float32]] = []

        for p in paths:
            try:
                y, _ = librosa.load(p, sr=self.sr, mono=True)
            except Exception as e:
                warnings.warn(f"ConcatSynth: skipping '{p}': {e}")
                continue
            n = len(y) // unit_samps
            if n < 1:
                warnings.warn(
                    f"ConcatSynth: skipping '{p}' — shorter than "
                    f"unit_size={self.unit_size} ms"
                )
                continue
            for i in range(n):
                unit = y[i * unit_samps:(i + 1) * unit_samps].astype(np.float32)
                all_units.append(unit)
                all_feats.append(_extract(unit, self.sr, self.features))

        n_units = len(all_units)
        if n_units < 2:
            warnings.warn("ConcatSynth: corpus has fewer than 2 units — nothing to play")
            return

        print(f"ConcatSynth: analysing {n_units} units across {len(paths)} file(s)…")
        feat_matrix = np.stack(all_feats).astype(np.float32)

        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(feat_matrix)

        n_q = min(self.n_candidates, n_units)
        knn = NearestNeighbors(n_neighbors=n_q, algorithm='ball_tree')
        knn.fit(feat_scaled)

        feat_names: List[str] = []
        for name in self.features:
            fn = _EXTRACTORS.get(name)
            if fn is None:
                continue
            dim = len(fn(all_units[0], self.sr))
            feat_names.extend([name] if dim == 1 else [f"{name}_{i}" for i in range(dim)])

        # Flat source array for GranularSynth's renderer + per-unit start offsets
        source = np.concatenate(all_units)
        unit_offsets = np.arange(n_units, dtype=np.int64) * unit_samps

        self._source = source
        self._unit_offsets = unit_offsets
        self._feat_matrix = feat_matrix
        self._feat_names = feat_names
        self._scaler = scaler
        self._knn = knn
        self._n_units = n_units
        self._unit_samps = unit_samps

        print(f"ConcatSynth: ready — {n_units} units, {feat_matrix.shape[1]} features")

    # ------------------------------------------------------------------
    # Target control
    # ------------------------------------------------------------------

    def set_target_from_audio(self, y: npt.NDArray[np.float32]) -> None:
        """Extract features from *y* and use them as the playback target.

        Args:
            y: Mono audio samples at the device sample rate.
        """
        if len(y) < 512 or not self._feat_names:
            return
        vec = _extract(y, self.sr, self.features)
        if len(vec) == len(self._feat_names):
            self._target_q.put(dict(zip(self._feat_names, vec.tolist())))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _target_vector(self) -> Optional[npt.NDArray[np.float32]]:
        if not self.target or not self._feat_names:
            return None
        vec = [
            float(self.target[name]) if name in self.target
            else float(self._feat_matrix[:, i].mean())
            for i, name in enumerate(self._feat_names)
        ]
        return np.array(vec, dtype=np.float32)

    def _select_unit_idx(self) -> int:
        target_vec = self._target_vector()
        if target_vec is not None:
            scaled = self._scaler.transform(target_vec.reshape(1, -1))
            n_q = min(self.n_candidates, self._n_units)
            if n_q < 1:
                n_q = 1
            _, indices = self._knn.kneighbors(scaled, n_q)
            return int(np.random.choice(indices[0]))
        return int(np.random.randint(0, self._n_units))

    # ------------------------------------------------------------------
    # GranularSynth overrides
    # ------------------------------------------------------------------

    def _spawn_grain(self, voice: dict) -> None:
        """Select a grain via KNN instead of position+spread in source file."""
        source = self._source
        if source is None or self._knn is None or len(source) < 2:
            return
        if len(self._grains) >= self.n_grains:
            return

        source_len = len(source)
        
        grain_samps = max(int(self.grain_size * self.sr / 1000.0), 2)

        # KNN unit selection
        unit_idx = self._select_unit_idx()
        unit_start = int(self._unit_offsets[unit_idx])
        # print(unit_idx, unit_start)

        # spread = scatter within the selected unit (0 = unit start, 1 = full unit)
        if self.spread > 0.0 and self._unit_samps > grain_samps:
            scatter_range = int(self.spread * (self._unit_samps - grain_samps))
            offset = int(np.random.uniform(0, scatter_range)) if scatter_range > 0 else 0
        else:
            offset = 0

        src_pos = float(np.clip(unit_start + offset, 0, source_len - grain_samps - 1))

        semitones = self.pitch
        if self.pitch_spread > 0.0:
            semitones += float(np.random.normal(0.0, self.pitch_spread))
        rate = max((2.0 ** (semitones / 12.0)) * (voice['freq'] / 440.0), 0.01)
        # print(rate, self.pitch, self.pitch_spread)
        self._grains.append({
            'source_phase': src_pos,
            'rate': rate,
            'vel': voice['vel'],
            'envelope': self._build_envelope(grain_samps),
            'phase': 0,
        })

    def audio_callback(self) -> npt.NDArray[np.float32]:
        # Drain target updates before the granular machinery runs
        while not self._target_q.empty():
            try:
                self.target.update(self._target_q.get_nowait())
            except queue.Empty:
                break
        return super().audio_callback()
