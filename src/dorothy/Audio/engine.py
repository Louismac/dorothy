"""
Main Audio engine class.
"""

import os
import warnings
from typing import Optional, Callable, List

import numpy as np
import numpy.typing as npt
import librosa
import sounddevice as sd

from .config import AudioConfig
from .device import AudioDevice
from .players import SamplePlayer, CustomPlayer, AudioCapture
from .ml_players import MAGNetPlayer, RAVEPlayer, ClusterResult, TORCH_AVAILABLE
from .synth import Note, SynthVoice, PolySynth
from .sequencer import Clock, Sequence
from .sampler import Sampler


class Audio:
    """Main class for music analysis and generation."""

    def __init__(self):
        """Initialize the Audio engine."""
        self.audio_outputs: List[AudioDevice] = []
        self.clocks: List[Clock] = []
        print("Loading Audio Engine")
        print(sd.query_devices())

    def start_magnet_stream(
        self,
        model_path: str,
        dataset_path: str,
        buffer_size: int = AudioConfig.DEFAULT_BUFFER_SIZE,
        sr: int = AudioConfig.DEFAULT_SAMPLE_RATE,
        output_device: Optional[int] = None
    ) -> int:
        """
        Start stream generating from a pretrained MAGNet model.

        Args:
            model_path: Path to the pretrained model
            dataset_path: Path to seed audio file used to train model
            buffer_size: Size of buffer when playing back / analysing audio
            sr: Sample rate to capture at
            output_device: Where to play this back to. Use sd.query_devices() to see available

        Returns:
            Index of the device in the audio_outputs list
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for MAGNet streaming")

        device = MAGNetPlayer(
            model_path,
            dataset_path,
            buffer_size=buffer_size,
            sr=sr,
            output_device=output_device
        )
        return self._register_and_play(device)

    def start_rave_stream(
        self,
        model_path: str = "",
        fft_size: int = AudioConfig.DEFAULT_FFT_SIZE,
        buffer_size: int = AudioConfig.DEFAULT_BUFFER_SIZE,
        sr: int = AudioConfig.DEFAULT_SAMPLE_RATE,
        latent_dim: int = 16,
        output_device: Optional[int] = None
    ) -> int:
        """
        Start stream generating from a pretrained RAVE model.

        Args:
            model_path: Path to the pretrained model
            fft_size: Size of FFT
            buffer_size: Size of buffer when playing back / analysing audio
            sr: Sample rate to capture at
            latent_dim: Number of latent dimensions (must match pretrained model)
            output_device: Where to play this back to

        Returns:
            Index of the device in the audio_outputs list
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for RAVE streaming")

        device = RAVEPlayer(
            model_path=model_path,
            buffer_size=buffer_size,
            sr=sr,
            fft_size=fft_size,
            latent_dim=latent_dim,
            output_device=output_device
        )
        return self._register_and_play(device)

    def update_rave_from_stream(self, input_idx: int = 0) -> None:
        """
        Use a given stream (e.g. file player or mic input) as input to RAVE streams.

        Args:
            input_idx: Index in audio_outputs to use as input
        """
        if input_idx >= len(self.audio_outputs):
            raise IndexError(f"Invalid input index: {input_idx}")

        input_device = self.audio_outputs[input_idx]

        def internal_callback():
            import torch
            with torch.no_grad():
                input_audio = input_device.audio_buffer
                for device in self.audio_outputs:
                    if isinstance(device, RAVEPlayer):
                        device.update_z(input_audio)

        input_device.gain = 0
        input_device.analyse = False
        input_device.internal_callback = internal_callback

    def start_device_stream(
        self,
        device: int,
        fft_size: int = AudioConfig.DEFAULT_FFT_SIZE,
        buffer_size: int = AudioConfig.DEFAULT_BUFFER_SIZE,
        sr: int = AudioConfig.DEFAULT_SAMPLE_RATE,
        analyse: bool = True
    ) -> int:
        """
        Start stream capturing audio from an input device.

        Args:
            device: Device ID to capture from (use sd.query_devices() to see available)
            fft_size: Size of FFT
            buffer_size: Size of buffer
            sr: Sample rate to capture at
            analyse: Whether to analyse for amplitude, FFT, etc.

        Returns:
            Index of the device in the audio_outputs list
        """
        audio_device = AudioCapture(
            analyse=analyse,
            buffer_size=buffer_size,
            sr=sr,
            fft_size=fft_size,
            input_device=device
        )
        return self._register_and_play(audio_device)

    def start_file_stream(
        self,
        file_path: str,
        fft_size: int = 512,
        buffer_size: int = 1024,
        sr: int = AudioConfig.DEFAULT_SAMPLE_RATE,
        output_device: Optional[int] = None,
        analyse: bool = True
    ) -> int:
        """
        Start stream of a given audio file.

        Args:
            file_path: Path to the audio file
            fft_size: Size of FFT
            buffer_size: Size of buffer
            sr: Sample rate of the audio
            output_device: Where to play this back to
            analyse: Whether to analyse for amplitude, FFT, etc.

        Returns:
            Index of the device in the audio_outputs list
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Load file
        y, loaded_sr = librosa.load(file_path, sr=sr, mono=False)
        return self.start_sample_stream(y, fft_size, buffer_size, loaded_sr, output_device, analyse)

    def start_dsp_stream(
        self,
        audio_callback: Callable[[int], npt.NDArray[np.float32]],
        fft_size: int = 512,
        buffer_size: int = 2048,
        sr: int = AudioConfig.DEFAULT_SAMPLE_RATE,
        output_device: Optional[int] = None,
        analyse: bool = True
    ) -> int:
        """
        Start stream with custom DSP callback for audio generation.

        Args:
            audio_callback: Callback function generating the audio
            fft_size: Size of FFT
            buffer_size: Size of buffer
            sr: Sample rate
            output_device: Where to play this back to
            analyse: Whether to analyse for amplitude, FFT, etc.

        Returns:
            Index of the device in the audio_outputs list
        """
        device = CustomPlayer(
            audio_callback,
            frame_size=buffer_size,
            analyse=analyse,
            fft_size=fft_size,
            buffer_size=buffer_size,
            sr=sr,
            output_device=output_device
        )
        return self._register_and_play(device)

    def start_poly_synth_stream(
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
        buffer_size: int = 512,
        sr: int = AudioConfig.DEFAULT_SAMPLE_RATE,
        output_device: Optional[int] = None,
        analyse: bool = True,
    ) -> int:
        """Start a polyphonic synthesizer stream.

        Returns the index of the new device in ``audio_outputs``.  Retrieve
        the :class:`PolySynth` instance via ``audio.audio_outputs[idx]`` to
        call ``note_on`` / ``note_off`` directly, or connect a
        :class:`Sequence` to drive it from a :class:`Clock`.

        Args:
            n_voices:    Maximum simultaneous voices (default 8).
            n_harmonics: Harmonics per voice (sine mode only, default 4).
            attack:      ADSR attack time in seconds.
            decay:       ADSR decay time in seconds.
            sustain:     ADSR sustain level 0-1.
            release:     ADSR release time in seconds.
            waveform:    Default oscillator: ``'sine'`` | ``'saw'`` |
                         ``'triangle'`` | ``'noise'`` | ``'supersaw'`` |
                         ``'fm'`` | ``'pwm'``.
            fm_ratio:    FM modulator frequency as multiple of carrier (default 2.0).
            fm_index:    FM modulation depth in radians (default 1.0).
            detune:      Supersaw total semitone spread (default 0.2).
            n_oscs:      Supersaw oscillator count (default 7).
            pwm:         PWM duty cycle 0-1 (default 0.5 = square).
            buffer_size: Audio buffer size in samples (smaller = lower latency).
            sr:          Sample rate.
            output_device: Output device index (None = system default).
            analyse:     Enable amplitude / FFT / onset / beat analysis.

        Returns:
            Index of the device in ``audio_outputs``.
        """
        device = PolySynth(
            n_voices=n_voices,
            n_harmonics=n_harmonics,
            attack=attack,
            decay=decay,
            sustain=sustain,
            release=release,
            waveform=waveform,
            fm_ratio=fm_ratio,
            fm_index=fm_index,
            detune=detune,
            n_oscs=n_oscs,
            pwm=pwm,
            analyse=analyse,
            buffer_size=buffer_size,
            sr=sr,
            output_device=output_device,
        )
        return self._register_and_play(device)

    def start_sample_stream(
        self,
        y: npt.NDArray[np.float32],
        fft_size: int = AudioConfig.DEFAULT_FFT_SIZE,
        buffer_size: int = AudioConfig.DEFAULT_BUFFER_SIZE,
        sr: int = AudioConfig.DEFAULT_SAMPLE_RATE,
        output_device: Optional[int] = None,
        analyse: bool = True
    ) -> int:
        """
        Start stream of given audio samples.

        Args:
            y: Audio samples as numpy array
            fft_size: Size of FFT
            buffer_size: Size of buffer
            sr: Sample rate of the audio
            output_device: Where to play this back to
            analyse: Whether to analyse for amplitude, FFT, etc.

        Returns:
            Index of the device in the audio_outputs list
        """
        y = np.array(y, dtype=np.float32)

        # Calculate beat information
        to_track = y if y.ndim == 1 else y[0, :]

        device = SamplePlayer(
            y=y,
            analyse=analyse,
            fft_size=fft_size,
            buffer_size=buffer_size,
            sr=sr,
            output_device=output_device
        )
        return self._register_and_play(device)

    def get_clock(self, bpm: int = 120) -> 'Clock':
        """
        Create and return a new Clock instance.

        Args:
            bpm: Beats per minute

        Returns:
            Clock instance
        """
        clock = Clock()
        clock.set_bpm(bpm)
        self.clocks.append(clock)
        return clock

    def fft(self, output: int = 0) -> npt.NDArray[np.float32]:
        """
        Return current FFT values (compensated for audio latency).

        Args:
            output: Audio output index to check

        Returns:
            FFT values as numpy array
        """
        if output < len(self.audio_outputs):
            device = self.audio_outputs[output]
            ptr = (device.audio_buffer_write_ptr + 1) % device.audio_latency
            return device.fft_vals[ptr]
        return np.zeros((AudioConfig.DEFAULT_FFT_SIZE // 2) + 1)

    def amplitude(self, output: int = 0) -> float:
        """
        Return current amplitude (compensated for audio latency).

        Args:
            output: Audio output index to check

        Returns:
            Average amplitude
        """
        if output < len(self.audio_outputs):
            device = self.audio_outputs[output]
            ptr = (device.audio_buffer_write_ptr + 1) % device.audio_latency
            return device.amplitude[ptr]
        return 0.0

    def is_onset(self, output: int = 0) -> bool:
        """
        Check if there has been an onset since the last time this was called.
        This is a "consume" operation - it clears the flag after reading.

        Returns:
            bool: True if onset occurred since last check, False otherwise
        """
        if output >= len(self.audio_outputs):
            return False
        device = self.audio_outputs[output]
        ptr = 0
        if device._onset_flag[ptr]:
            device._onset_flag[ptr] = False
            return True
        return False

    def is_beat(self, output: int = 0) -> bool:
        """
        Check if there has been a beat since last called.

        Args:
            output: Audio output index to check

        Returns:
            True if beat detected, False otherwise
        """
        if output >= len(self.audio_outputs):
            return False
        device = self.audio_outputs[output]
        ptr = 0
        if device._beat_flag[ptr]:
            device._beat_flag[ptr] = False
            return True
        else:
            return False

    def play(self, output: int = 0) -> None:
        """Start playback on specified output."""
        if output < len(self.audio_outputs):
            self.audio_outputs[output].play()

    def stop(self, output: int = 0) -> None:
        """Stop playback on specified output."""
        if output < len(self.audio_outputs):
            self.audio_outputs[output].stop()

    def pause(self, output: int = 0) -> None:
        """Pause playback on specified output."""
        if output < len(self.audio_outputs):
            self.audio_outputs[output].pause()

    def resume(self, output: int = 0) -> None:
        """Resume playback on specified output."""
        if output < len(self.audio_outputs):
            self.audio_outputs[output].resume()

    def clean_up(self) -> None:
        """Clean up all audio outputs and clocks."""
        for device in self.audio_outputs:
            try:
                device.stop()
            except Exception as e:
                warnings.warn(f"Error stopping device: {e}")

        for clock in self.clocks:
            try:
                clock.stop()
            except Exception as e:
                warnings.warn(f"Error stopping clock: {e}")

    def _register_and_play(self, device: 'AudioDevice') -> int:
        """Register device and start playback."""
        self.audio_outputs.append(device)
        index = len(self.audio_outputs) - 1
        print("playing", index)
        self.play(index)
        return index
