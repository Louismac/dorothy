"""
Configuration constants for the Dorothy Audio Engine.
"""

from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Configuration constants for audio processing."""
    DEFAULT_AUDIO_LATENCY: int = 5
    DEFAULT_FFT_SIZE: int = 1024
    DEFAULT_BUFFER_SIZE: int = 2048
    DEFAULT_SAMPLE_RATE: int = 44100
    MAGNET_FRAME_SIZE: int = 1024 * 75
    RAVE_FRAME_SIZE: int = 2048
    MAX_RECORDING_DURATION: int = 600000  # milliseconds
    THREAD_JOIN_TIMEOUT: float = 2.0
