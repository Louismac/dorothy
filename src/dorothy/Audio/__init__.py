"""
Dorothy Audio package - drop-in replacement for the original Audio.py module.

Re-exports every public name so that existing imports such as:
    from .Audio import Audio
    from .Audio import Note, Sequence, Clock
    from .Audio import AudioDevice, SamplePlayer, CustomPlayer, Sampler
    from .Audio import PolySynth, SynthVoice
    from .Audio import StreamingOnsetDetector, StreamingBeatTracker
    from .Audio import MAGNetPlayer, RAVEPlayer, ClusterResult
    from .Audio import AudioConfig
all continue to work without modification.
"""

from .config import AudioConfig

from .analysis import StreamingOnsetDetector, StreamingBeatTracker

from .device import AudioDevice

from .players import SamplePlayer, CustomPlayer, AudioCapture

from .ml_players import MAGNetPlayer, RAVEPlayer, ClusterResult, TORCH_AVAILABLE

from .synth import Note, SynthVoice, PolySynth

from .sequencer import Clock, Sequence

from .sampler import Sampler

from .engine import Audio

__all__ = [
    "AudioConfig",
    "StreamingOnsetDetector",
    "StreamingBeatTracker",
    "AudioDevice",
    "SamplePlayer",
    "CustomPlayer",
    "AudioCapture",
    "MAGNetPlayer",
    "RAVEPlayer",
    "ClusterResult",
    "TORCH_AVAILABLE",
    "Note",
    "SynthVoice",
    "PolySynth",
    "Clock",
    "Sequence",
    "Sampler",
    "Audio",
]
