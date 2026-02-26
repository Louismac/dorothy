"""
Audio player classes: SamplePlayer, CustomPlayer, AudioCapture.
"""

import time
import threading
import warnings
from typing import Optional

import numpy as np
import numpy.typing as npt
import librosa
import sounddevice as sd

from .config import AudioConfig
from .device import AudioDevice


class SamplePlayer(AudioDevice):
    """Play back audio samples"""

    def __init__(
        self,
        y: npt.NDArray[np.float32],
        **kwargs
    ):
        """Initialize sample player."""
        super().__init__(use_streaming_analysis=False, **kwargs)

        # Ensure audio is 2D
        self.y = y if y.ndim == 2 else y[np.newaxis, :]
        if self.y is not None:
            # Calculate beat information
            to_track = self.y if self.y.ndim == 1 else self.y[0, :]
            self.tempo, self.beats = librosa.beat.beat_track(y=to_track, sr=self.sr, units='samples')
            # Compute the onset strength envelope
            onset_env = librosa.onset.onset_strength(y=to_track, sr=self.sr)
            self.onsets = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                y=to_track,
                sr=self.sr,
                units="samples", delta=0.2)
            print(f"found {len(self.beats)} beats and {len(self.onsets)} onsets")
        self.current_sample = 0
        self.beat_ptr = 0
        self.onset_ptr = 0

    def check_onset(self, *args, **kwargs) -> bool:
        """Check if an onset has occurred since last call."""
        if len(self.onsets) == 0 or self.onset_ptr >= len(self.onsets):
            return False

        next_onset = self.onsets[self.onset_ptr]
        if next_onset < self.current_sample:
            self.onset_ptr += 1
            return True

        return False

    def check_beat(self) -> bool:
        """Check if a beat has occurred since last call."""
        if len(self.beats) == 0 or self.beat_ptr >= len(self.beats):
            return False

        next_beat = self.beats[self.beat_ptr]

        if next_beat < self.current_sample:
            self.beat_ptr += 1
            return True

        return False

    def reset_ptrs(self):
        self.onset_ptr = 0
        self.beat_ptr = 0

    def on_loop(self, audio_buffer):
        self.reset_ptrs()
        wrap_ptr = self.current_sample - self.y.shape[1]
        wrap_signal = self.y[:, :wrap_ptr]
        audio_buffer = np.concatenate((audio_buffer, wrap_signal), axis=1)
        self.current_sample = wrap_ptr
        return audio_buffer

    def audio_callback(self) -> npt.NDArray[np.float32]:
        """Generate audio samples from stored audio."""
        if self.pause_event.is_set():
            return self._get_silence()

        # Get audio buffer
        audio_buffer = self.y[:, self.current_sample:self.current_sample + self.buffer_size]

        # Advance playhead
        self.current_sample += self.buffer_size

        # Handle wrapping (if we go off the end)
        if self.current_sample >= self.y.shape[1]:
            audio_buffer = self.on_loop(audio_buffer)

        # Store for analysis (use first channel)
        self.audio_buffer = audio_buffer[0, :] if audio_buffer.shape[0] > 0 else audio_buffer

        self.internal_callback()
        self.on_new_frame(audio_buffer)
        return audio_buffer * self.gain


class CustomPlayer(AudioDevice):
    def __init__(self, get_frame, frame_size=512, **kwargs):
        super().__init__(**kwargs)
        self.frame_size = frame_size
        self.get_frame = get_frame
        self.current_sample = 0
        self.current_buffer = np.zeros(self.frame_size, dtype=np.float32)
        self.next_buffer = np.zeros(self.frame_size, dtype=np.float32)
        self.generate_thread = threading.Thread(target=self.fill_next_buffer)
        self.generate_thread.start()

    def fill_next_buffer(self):
        if self.get_frame is not None:
            self.next_buffer = self.get_frame(self.frame_size).astype(np.float32)

    def audio_callback(self):
        if self.pause_event.is_set():
            print("paused")
            return np.zeros((self.channels, self.buffer_size), dtype=np.float32)  # Fill buffer with silence if paused
        else:
            audio_buffer = self.current_buffer[self.current_sample:self.current_sample + self.buffer_size]
            self.current_sample += self.buffer_size
            if self.current_sample >= self.frame_size:
                self.current_sample = 0
                self.current_buffer = self.next_buffer.copy()
                if not self.generate_thread.is_alive():
                    self.generate_thread = threading.Thread(target=self.fill_next_buffer)
                    self.generate_thread.start()
            self.internal_callback()
            self.on_new_frame(audio_buffer)
            return audio_buffer * self.gain


class AudioCapture(AudioDevice):
    """Capture and analyze audio streams in real-time."""

    def __init__(self, input_device: Optional[int] = None, **kwargs):
        """Initialize audio capture."""
        super().__init__(**kwargs)
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.input_device = input_device

    def audio_callback(self, indata: npt.NDArray[np.float32], frames: int,
                       time_info: dict, status: sd.CallbackFlags) -> None:
        """Process incoming audio data."""
        if status:
            warnings.warn(f"Audio callback status: {status}")

        if self.pause_event.is_set():
            return

        try:
            # Store first channel
            self.audio_buffer = indata[:, 0]

            # Run callbacks
            self.internal_callback()
            self.do_analysis(self.audio_buffer)
            self.on_new_frame(self.audio_buffer)

            # Handle recording
            if self.recording:
                self._handle_recording_input(indata)

        except Exception as e:
            import traceback
            traceback.print_exception(e)
            warnings.warn(f"Audio capture error: {e}")

    def _handle_recording_input(self, audio_data: npt.NDArray[np.float32]) -> None:
        """Handle recording for input streams."""
        elapsed = time.time() - self._recording_start_time
        if elapsed > AudioConfig.MAX_RECORDING_DURATION:
            self.stop_recording()
            print("Maximum recording duration reached")
        else:
            # Ensure stereo output
            channels = 2
            if audio_data.ndim == 1:
                audio_data = audio_data[:, np.newaxis]

            # Match channel count
            current_channels = audio_data.shape[1]
            if current_channels < channels:
                audio_data = np.tile(audio_data, (1, channels))
            elif current_channels > channels:
                audio_data = audio_data[:, :channels]

            self.recording_buffer.append(audio_data.copy())

    def capture_audio(self) -> None:
        """Main audio capture loop for input streams."""
        if self.input_device is None:
            self.input_device = sd.default.device[0]
            print(f"Input device set to default: {sd.default.device[0]}")

        device_info = sd.query_devices(self.input_device)
        self.channels = min(2, device_info['max_input_channels'])
        print(f"Channels: {self.channels}")

        print(f"Capturing audio: device={self.input_device}, channels={self.channels}")

        try:
            with sd.InputStream(
                callback=self.audio_callback,
                channels=self.channels,
                blocksize=self.buffer_size,
                samplerate=self.sr,
                device=self.input_device
            ) as stream:
                self.stream = stream
                while self.running and not self._shutdown_event.is_set():
                    time.sleep(0.1)
        except Exception as e:
            warnings.warn(f"Capture stream error: {e}")
        finally:
            self.running = False
