"""
Sampler class for triggering audio samples.
"""

import warnings
from typing import TYPE_CHECKING, Any, List

import numpy as np
import numpy.typing as npt
import librosa

if TYPE_CHECKING:
    from .engine import Audio


class Sampler:
    """Simple sampler for triggering audio samples."""

    def __init__(self, audio_instance: Any):
        """
        Initialize sampler.

        Args:
            audio_instance: The Audio instance to attach to
        """
        self.samples: List[npt.NDArray[np.float32]] = [np.zeros(1024, dtype=np.float32)]
        self.sample_pos: List[int] = [-1]
        self.audio = audio_instance

        # Create DSP callback
        def get_frame(size: int) -> npt.NDArray[np.float32]:
            audio = np.zeros(size, dtype=np.float32)
            for i, p in enumerate(self.sample_pos):
                if p >= 0:
                    end = p + size
                    sample = self.samples[i]

                    if end >= len(sample):
                        remaining = len(sample) - p
                        audio[:remaining] += sample[p:p + remaining]
                        self.sample_pos[i] = -1
                    else:
                        audio += sample[p:end]
                        self.sample_pos[i] += size

            return audio

        # Start DSP stream
        self.audio.start_dsp_stream(get_frame, sr=22050, buffer_size=512)

    def trigger(self, index: int) -> None:
        """
        Trigger playback of a sample.

        Args:
            index: Index of the sample to trigger
        """
        if 0 <= index < len(self.samples):
            self.sample_pos[index] = 0

    def load(self, paths: List[str]) -> None:
        """
        Load audio samples from file paths.

        Args:
            paths: List of file paths to load
        """
        self.samples = []
        for path in paths:
            try:
                y, _ = librosa.load(path, sr=22050)
                self.samples.append(y)
            except Exception as e:
                warnings.warn(f"Could not load sample {path}: {e}")
                self.samples.append(np.zeros(1024, dtype=np.float32))

        self.sample_pos = [-1] * len(self.samples)
