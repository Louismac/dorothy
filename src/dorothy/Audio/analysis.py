"""
Real-time audio analysis: onset detection and beat tracking.
"""

import time
import warnings
from collections import deque
from typing import Optional

import numpy as np
import numpy.typing as npt


class StreamingOnsetDetector:
    """
    Real-time onset detection using spectral flux with adaptive thresholding.
    Integrates with existing FFT analysis for efficiency.
    """

    def __init__(
        self,
        sample_rate=44100,
        fft_size=2048,
        hop_length=512,
        threshold=0.3,
        n_bands=6,
        wait=20
    ):
        """
        Args:
            sample_rate: Audio sample rate
            fft_size: FFT size (must match your existing analysis)
            hop_length: Hop size for analysis (must match your existing analysis)
            n_bands: Number of frequency bands for multi-band analysis
            threshold: Base threshold multiplier for onset detection
            pre_max: Frames before peak that must be lower (peak picking)
            post_max: Frames after peak that must be lower
            pre_avg: Frames for pre-average in adaptive threshold
            post_avg: Frames for post-average in adaptive threshold
            delta: Constant added to adaptive threshold
            wait: Minimum frames between consecutive onsets
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.n_bands = n_bands
        self.threshold = threshold
        self.pre_max = int(0.03 * self.sample_rate // self.hop_length)  # 30ms
        self.post_max = int(0.00 * self.sample_rate // self.hop_length + 1)  # 0ms
        self.pre_avg = int(0.10 * self.sample_rate // self.hop_length)  # 100ms
        self.post_avg = int(0.10 * self.sample_rate // self.hop_length + 1)  # 100ms
        self.wait = int(0.03 * self.sample_rate // self.hop_length)  # 30ms
        self.delta = 0.07

        # State for onset detection
        self.prev_magnitude = None

        # Onset detection function (ODF) history
        max_history = max(self.pre_max + self.post_max, self.pre_avg + self.post_avg) + 10
        self.odf_history = deque(maxlen=max_history)

        # Onset timestamps (in samples)
        self.onset_positions = deque(maxlen=1000)
        self.total_samples_processed = 0
        self.frames_since_onset = wait  # Start ready to detect

        # Create filterbank for multi-band analysis
        self._create_filterbank()

    def _create_filterbank(self):
        """Create triangular filterbank for multi-band analysis."""
        # Frequency bands (Hz): exponentially spaced
        self.band_edges = np.logspace(
            np.log10(50),  # Start at 50 Hz
            np.log10(self.sample_rate / 2),  # Up to Nyquist
            self.n_bands + 1
        )

        # Convert to FFT bins
        self.band_bins = (self.band_edges * self.fft_size / self.sample_rate).astype(int)

    def _compute_spectral_flux(self, magnitude_spectrum):
        """
        Compute multi-band spectral flux with half-wave rectification.
        This is the core onset detection function (ODF).

        Args:
            magnitude_spectrum: FFT magnitude (from your do_analysis function)
        """
        if self.prev_magnitude is None:
            self.prev_magnitude = magnitude_spectrum.copy()
            return 0.0

        # Half-wave rectified spectral difference (only increases matter)
        diff = magnitude_spectrum - self.prev_magnitude
        diff = np.maximum(diff, 0)  # Half-wave rectification

        # Multi-band processing (emphasizes different frequency regions)
        band_fluxes = []
        for i in range(self.n_bands):
            start_bin = self.band_bins[i]
            end_bin = self.band_bins[i + 1]
            band_flux = np.sum(diff[start_bin:end_bin])
            band_fluxes.append(band_flux)

        # Weight higher frequencies more (transients often have high-freq content)
        weights = np.linspace(1.0, 2.0, self.n_bands)
        flux = np.sum(np.array(band_fluxes) * weights)

        self.prev_magnitude = magnitude_spectrum.copy()
        return flux

    def _adaptive_threshold(self, odf_value):
        """
        Adaptive threshold based on local average.
        Returns True if odf_value exceeds the adaptive threshold.
        """
        if len(self.odf_history) < self.pre_avg + self.post_avg:
            return False

        # Get local neighborhood
        history_list = list(self.odf_history)
        pre_window = history_list[-self.pre_avg:]
        post_window = history_list[-(self.pre_avg + self.post_avg):-self.pre_avg]

        # Adaptive threshold: mean of surrounding values + delta
        threshold_value = (
            self.threshold * (np.mean(pre_window) + np.mean(post_window)) / 2 + self.delta
        )

        return odf_value > threshold_value

    def _is_local_maximum(self):
        """Peak picking: check if current position is a local maximum."""
        if len(self.odf_history) < self.pre_max + self.post_max + 1:
            return False

        history_list = list(self.odf_history)
        current = history_list[-self.post_max - 1]

        # Check pre-max window
        pre_window = history_list[-(self.pre_max + self.post_max + 1):-self.post_max - 1]
        if any(v >= current for v in pre_window):
            return False

        # Check post-max window
        post_window = history_list[-self.post_max:]
        if any(v >= current for v in post_window):
            return False

        return True

    def process_fft_frame(self, magnitude_spectrum, num_frames=1):
        """
        Process FFT magnitude for onset detection.
        Call this from within your do_analysis function.

        Args:
            magnitude_spectrum: FFT magnitude array from your analysis
            num_frames: Number of FFT frames represented (for accumulated FFTs)

        Returns:
            bool: True if onset detected in this frame
        """
        # Compute onset detection function
        odf_value = self._compute_spectral_flux(magnitude_spectrum)
        self.odf_history.append(odf_value)

        # Increment frame counter
        self.frames_since_onset += 1

        # Detect onset (with all conditions)
        onset_detected = False
        if (self.frames_since_onset >= self.wait and
                self._adaptive_threshold(odf_value) and
                self._is_local_maximum()):

            # Record onset position (in samples)
            onset_sample = self.total_samples_processed
            self.onset_positions.append(onset_sample)
            self.frames_since_onset = 0
            onset_detected = True

        # Update sample counter (account for multiple frames if averaged)
        self.total_samples_processed += self.hop_length * num_frames
        return onset_detected

    def has_onset_in_range(self, n_samples):
        """
        Check if there was an onset in the last n_samples.

        Args:
            n_samples: Look back this many samples

        Returns:
            bool: True if onset detected in range
        """
        if not self.onset_positions:
            return False

        cutoff = self.total_samples_processed - n_samples
        return self.onset_positions[-1] >= cutoff

    def get_onsets_in_range(self, n_samples):
        """
        Get all onset positions within the last n_samples.

        Args:
            n_samples: Look back this many samples

        Returns:
            list: Sample positions of onsets (relative to start of stream)
        """
        if not self.onset_positions:
            return []

        cutoff = self.total_samples_processed - n_samples
        return [pos for pos in self.onset_positions if pos >= cutoff]

    def get_time_since_last_onset(self):
        """
        Get time in seconds since last onset.

        Returns:
            float: Seconds since last onset, or None if no onsets yet
        """
        if not self.onset_positions:
            return None

        samples_since = self.total_samples_processed - self.onset_positions[-1]
        return samples_since / self.sample_rate

    def reset(self):
        """Reset detector state."""
        self.prev_magnitude = None
        self.odf_history.clear()
        self.onset_positions.clear()
        self.total_samples_processed = 0
        self.frames_since_onset = self.wait


class StreamingBeatTracker:
    """
    Real-time beat tracking with tempo stability and lock-in.
    """

    def __init__(
        self,
        sample_rate=44100,
        tempo_range=(60, 180),
        tempo_resolution=2.0,
        tempo_weight_decay=0.85,
        beat_threshold=0.15,
        max_onsets=50,
        phase_tolerance=0.20,
        min_onsets_for_tracking=4,
        confidence_decay=0.98,
        tempo_lock_threshold=0.6,  # NEW: Confidence needed to lock tempo
        tempo_lock_variance=3.0,   # NEW: Max BPM variance when locked
    ):
        self.sample_rate = sample_rate
        self.tempo_range = tempo_range
        self.tempo_resolution = tempo_resolution
        self.tempo_weight_decay = tempo_weight_decay
        self.beat_threshold = beat_threshold
        self.max_onsets = max_onsets
        self.phase_tolerance = phase_tolerance
        self.min_onsets_for_tracking = min_onsets_for_tracking
        self.confidence_decay = confidence_decay
        self.tempo_lock_threshold = tempo_lock_threshold
        self.tempo_lock_variance = tempo_lock_variance

        # Create tempo hypothesis grid
        self.tempo_grid = np.arange(
            tempo_range[0],
            tempo_range[1] + tempo_resolution,
            tempo_resolution
        )
        self.n_tempos = len(self.tempo_grid)

        # Convert BPM to seconds per beat
        self.period_grid = 60.0 / self.tempo_grid

        # Tempo probability distribution
        self.tempo_probs = np.ones(self.n_tempos, dtype=np.float32) / self.n_tempos

        # Onset history
        self.onset_times = deque(maxlen=max_onsets)

        # Beat tracking state
        self.last_beat_time = None
        self.predicted_beat_time = None
        self.current_tempo_bpm = None
        self.beat_confidence = 0.0
        self.base_confidence = 0.0

        # Beats without onset support
        self.beats_since_onset = 0

        # NEW: Tempo stability tracking
        self.tempo_history = deque(maxlen=20)  # Longer history
        self.tempo_locked = False
        self.locked_tempo_bpm = None
        self.beats_at_locked_tempo = 0

        # Active tracking flag
        self.is_tracking = False
        self.start_time = time.time()

    def add_onset(self) -> None:
        """Add a new onset time to the tracker."""
        onset_time_seconds = time.time() - self.start_time
        self.onset_times.append(onset_time_seconds)

        # Update tempo estimates with new onset
        if len(self.onset_times) >= 2:
            self._update_tempo_probabilities()

            # Only start beat tracking after enough onsets
            if len(self.onset_times) >= self.min_onsets_for_tracking:
                self._update_beat_prediction()
                self.is_tracking = True

        # Check if onset supports current beat prediction
        if self.is_tracking and self.predicted_beat_time is not None:
            self._check_onset_support(onset_time_seconds)

    def _check_onset_support(self, onset_time: float) -> None:
        """Check if an onset aligns with the current beat prediction."""
        if self.last_beat_time is None or self.current_tempo_bpm is None:
            return

        beat_period = 60.0 / self.current_tempo_bpm

        # Calculate phase of this onset relative to beat grid
        time_since_beat = onset_time - self.last_beat_time
        phase_in_beat = (time_since_beat % beat_period) / beat_period

        # Check if onset is close to a beat
        alignment = min(phase_in_beat, 1.0 - phase_in_beat)

        # If onset aligns well with beat, boost confidence and reset counter
        if alignment < self.phase_tolerance:
            self.beats_since_onset = 0
            self.base_confidence = min(1.0, self.base_confidence * 1.1)
            self.beat_confidence = self.base_confidence

            # NEW: Track successful beats for tempo locking
            if self.tempo_locked:
                self.beats_at_locked_tempo += 1

    def _update_tempo_probabilities(self) -> None:
        """Update Bayesian tempo probability distribution."""
        if len(self.onset_times) < 2:
            return

        # NEW: If tempo is locked, constrain search around locked tempo
        if self.tempo_locked and self.locked_tempo_bpm is not None:
            # Only update probabilities near the locked tempo
            locked_period = 60.0 / self.locked_tempo_bpm
            search_mask = np.abs(self.period_grid - locked_period) < (self.tempo_lock_variance / 60.0)
        else:
            search_mask = np.ones(self.n_tempos, dtype=bool)

        # Decay previous probabilities
        self.tempo_probs *= self.tempo_weight_decay

        # Calculate recent inter-onset intervals
        onset_list = list(self.onset_times)
        recent_iois = []

        # Look at last several onsets
        lookback = min(10, len(onset_list))  # Increased lookback
        for i in range(len(onset_list) - lookback, len(onset_list)):
            if i > 0:
                ioi = onset_list[i] - onset_list[i - 1]
                if 0.2 < ioi < 3.0:
                    recent_iois.append(ioi)

        if not recent_iois:
            return

        # Better likelihood calculation
        new_probs = np.zeros(self.n_tempos, dtype=np.float32)

        for tempo_idx, period in enumerate(self.period_grid):
            # Skip if outside search mask (when locked)
            if not search_mask[tempo_idx]:
                continue

            likelihood = 0.0

            # Each IOI votes for tempos
            for ioi in recent_iois:
                # Check integer multiples/divisions
                # NEW: Only check 1.0x multiplier when locked (avoid octave confusion)
                if self.tempo_locked:
                    multipliers = [1.0]
                else:
                    multipliers = [0.5, 1.0, 2.0]

                for multiplier in multipliers:
                    expected_ioi = period * multiplier

                    # Gaussian likelihood
                    error = abs(ioi - expected_ioi)
                    sigma = 0.08
                    likelihood += np.exp(-(error ** 2) / (2 * sigma ** 2))

            new_probs[tempo_idx] = likelihood

        # NEW: Stronger smoothing when locked
        if self.tempo_locked:
            self.tempo_probs = 0.7 * self.tempo_probs + 0.3 * new_probs  # More conservative
        else:
            self.tempo_probs = 0.3 * self.tempo_probs + 0.7 * new_probs  # Favor new evidence

        # Normalize
        prob_sum = np.sum(self.tempo_probs)
        if prob_sum > 0:
            self.tempo_probs /= prob_sum

        # Sharpen distribution
        self.tempo_probs = np.power(self.tempo_probs, 1.5)
        prob_sum = np.sum(self.tempo_probs)
        if prob_sum > 0:
            self.tempo_probs /= prob_sum

        # Extract most likely tempo
        best_tempo_idx = np.argmax(self.tempo_probs)
        new_tempo_bpm = self.tempo_grid[best_tempo_idx]

        # NEW: Smooth tempo changes (don't jump around)
        if self.current_tempo_bpm is not None:
            # Use exponential moving average
            alpha = 0.3 if self.tempo_locked else 0.5
            self.current_tempo_bpm = (alpha * new_tempo_bpm +
                                      (1 - alpha) * self.current_tempo_bpm)
        else:
            self.current_tempo_bpm = new_tempo_bpm

        # Confidence calculation
        sorted_probs = np.sort(self.tempo_probs)[::-1]
        if len(sorted_probs) > 1 and sorted_probs[1] > 0:
            self.base_confidence = min(1.0, sorted_probs[0] / (sorted_probs[1] + 0.01))
        else:
            self.base_confidence = sorted_probs[0]

        # Track tempo stability
        self.tempo_history.append(self.current_tempo_bpm)

        # NEW: Check if we should lock onto this tempo
        if not self.tempo_locked and len(self.tempo_history) >= 8:
            tempo_std = np.std(list(self.tempo_history)[-8:])
            tempo_mean = np.mean(list(self.tempo_history)[-8:])

            # Lock if tempo is stable and confidence is high
            if tempo_std < 4.0 and self.base_confidence > self.tempo_lock_threshold:
                self.tempo_locked = True
                self.locked_tempo_bpm = tempo_mean
                self.beats_at_locked_tempo = 0
                print(f"TEMPO LOCKED: {self.locked_tempo_bpm:.1f} BPM (std: {tempo_std:.2f})")

        # NEW: Unlock if tempo has been consistently wrong
        if self.tempo_locked and self.beats_at_locked_tempo > 20:
            # Check if locked tempo is still valid
            recent_tempo_std = np.std(list(self.tempo_history)[-5:])
            if recent_tempo_std > 8.0:  # Tempo is drifting
                print(f"TEMPO UNLOCKED (drift detected: {recent_tempo_std:.2f})")
                self.tempo_locked = False
                self.locked_tempo_bpm = None
                self.beats_at_locked_tempo = 0

        # Increase confidence if tempo is stable
        if len(self.tempo_history) >= 5:
            tempo_std = np.std(list(self.tempo_history))
            if tempo_std < 5.0:
                self.base_confidence *= 1.5
                self.base_confidence = min(1.0, self.base_confidence)

        self.beat_confidence = self.base_confidence

    def _update_beat_prediction(self) -> None:
        """Update beat phase tracking and predict next beat time."""
        if len(self.onset_times) < self.min_onsets_for_tracking or self.current_tempo_bpm is None:
            return

        current_time = self.onset_times[-1]
        beat_period = 60.0 / self.current_tempo_bpm

        # Use recent anchor point instead of t=0
        onset_list = list(self.onset_times)
        recent_onsets = onset_list[-min(12, len(onset_list)):]
        anchor_time = np.median(recent_onsets)

        # Test different phase offsets relative to anchor
        n_phase_bins = 20
        phase_votes = np.zeros(n_phase_bins)

        for phase_bin in range(n_phase_bins):
            phase_offset = (phase_bin / n_phase_bins) * beat_period

            for onset_time in recent_onsets:
                time_since_anchor = onset_time - anchor_time
                time_in_beat_grid = (time_since_anchor - phase_offset) % beat_period
                alignment = min(time_in_beat_grid, beat_period - time_in_beat_grid)

                # Gaussian weighting
                sigma = self.phase_tolerance * beat_period
                vote_strength = np.exp(-(alignment ** 2) / (2 * sigma ** 2))
                phase_votes[phase_bin] += vote_strength

        # Choose best phase offset from anchor
        best_phase_bin = np.argmax(phase_votes)
        best_phase_offset = (best_phase_bin / n_phase_bins) * beat_period

        # The actual phase at anchor time
        phase_at_anchor = anchor_time + best_phase_offset

        # Predict next beat relative to current time
        time_since_phase = current_time - phase_at_anchor
        beats_since_phase = time_since_phase / beat_period
        next_beat_number = np.ceil(beats_since_phase)

        self.predicted_beat_time = phase_at_anchor + (next_beat_number * beat_period)

        # Track last beat
        if self.last_beat_time is None or abs(current_time - self.last_beat_time) > beat_period * 0.5:
            self.last_beat_time = phase_at_anchor + (np.floor(beats_since_phase) * beat_period)

    def check_beat(self) -> bool:
        current_time_seconds = time.time() - self.start_time
        """Check if a beat should occur at the current time."""
        if not self.is_tracking or self.predicted_beat_time is None:
            return False

        if self.beat_confidence < self.beat_threshold:
            return False

        # Check if we've reached the predicted beat time
        if current_time_seconds >= self.predicted_beat_time:
            # Beat occurred!
            self.last_beat_time = self.predicted_beat_time

            # Predict next beat
            beat_period = 60.0 / self.current_tempo_bpm
            self.predicted_beat_time += beat_period

            # Increment beats without onset support
            self.beats_since_onset += 1

            return True

        return False

    def get_next_beat_time(self) -> Optional[float]:
        """Get the predicted time of the next beat."""
        return self.predicted_beat_time

    def get_current_tempo(self) -> Optional[float]:
        """Get the current estimated tempo."""
        return self.current_tempo_bpm

    def get_beat_confidence(self) -> float:
        """Get confidence in current beat prediction (0-1)."""
        return self.beat_confidence

    def get_tempo_distribution(self) -> tuple:
        """
        Get the full tempo probability distribution for debugging.

        Returns:
            tuple: (tempo_grid, tempo_probabilities)
        """
        return self.tempo_grid, self.tempo_probs

    def get_time_to_next_beat(self, current_time_seconds: float) -> Optional[float]:
        """Get time remaining until next beat."""
        if self.predicted_beat_time is None:
            return None

        time_to_beat = self.predicted_beat_time - current_time_seconds
        return max(0.0, time_to_beat)

    def get_beat_phase(self, current_time_seconds: float) -> Optional[float]:
        """Get current position in beat cycle (0-1)."""
        if self.last_beat_time is None or self.current_tempo_bpm is None:
            return None

        beat_period = 60.0 / self.current_tempo_bpm
        time_since_beat = current_time_seconds - self.last_beat_time
        phase = (time_since_beat % beat_period) / beat_period

        return phase

    def reset(self) -> None:
        """Reset all tracking state."""
        self.tempo_probs = np.ones(self.n_tempos, dtype=np.float32) / self.n_tempos
        self.onset_times.clear()
        self.tempo_history.clear()
        self.last_beat_time = None
        self.predicted_beat_time = None
        self.current_tempo_bpm = None
        self.beat_confidence = 0.0
