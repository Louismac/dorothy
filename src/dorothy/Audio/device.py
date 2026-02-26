"""
AudioDevice base class for all audio providers.
"""

import os
import time
import threading
import ctypes
import warnings
from typing import Optional, Callable, List

import numpy as np
import numpy.typing as npt
import sounddevice as sd
import psutil

from .config import AudioConfig
from .analysis import StreamingOnsetDetector, StreamingBeatTracker


class AudioDevice:
    """Base class for all audio providers."""

    def __init__(
        self,
        on_new_frame: Callable[[npt.NDArray[np.float32]], None] = lambda x: None,
        analyse: bool = True,
        fft_size: int = AudioConfig.DEFAULT_FFT_SIZE,
        buffer_size: int = AudioConfig.DEFAULT_BUFFER_SIZE,
        sr: int = AudioConfig.DEFAULT_SAMPLE_RATE,
        output_device: Optional[int] = None,
        use_streaming_analysis: bool = True,
    ):
        """Initialize audio device."""
        # Validation
        if buffer_size <= 0 or fft_size <= 0:
            raise ValueError("Buffer and FFT sizes must be positive")
        if sr <= 0:
            raise ValueError("Sample rate must be positive")

        # Configuration
        self.sr = sr
        self.fft_size = fft_size
        self.buffer_size = buffer_size
        self.analyse = analyse
        self.analyse_fft = analyse
        # requires fft
        self.analyse_onsets = False
        # requires onsets
        self.analyse_beats = False
        self.output_device = output_device
        self.channels = 1
        self.gain = 1.0

        # State
        self.running = False
        self.recording = False
        self.audio_latency = AudioConfig.DEFAULT_AUDIO_LATENCY
        self.audio_buffer_write_ptr = 0

        # Threading
        self.pause_event = threading.Event()
        self._shutdown_event = threading.Event()
        self.play_thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_new_frame = on_new_frame
        self.internal_callback = lambda: None

        # Recording
        self.recording_buffer: List[npt.NDArray[np.float32]] = []
        self._recording_start_time = time.time()

        # Analysis buffers
        self._init_analysis_buffers()
        self._fft_window = np.hanning(self.fft_size)

        if use_streaming_analysis:
            self.onset_detector = StreamingOnsetDetector(
                sample_rate=self.sr,
                fft_size=fft_size,
                hop_length=self.hop_length,
                threshold=0.3,
            )
            self.beat_tracker = StreamingBeatTracker(
                sample_rate=self.sr,
                tempo_range=(60, 180),
                beat_threshold=0.5,  # Lower = more permissive
            )

        # Warn about potential issues
        if self.fft_size > self.buffer_size:
            warnings.warn(
                "FFT window is larger than buffer size. "
                "Zero-padding will occur, which may lead to unexpected results.",
                UserWarning
            )

        # Set process priority
        self._set_process_priority()

        # Set default sample rate
        sd.default.samplerate = self.sr

    def _init_analysis_buffers(self) -> None:
        """Initialize analysis buffers."""
        self.fft_vals = [
            np.zeros((self.fft_size // 2) + 1, dtype=np.float32)
            for _ in range(self.audio_latency)
        ]
        self.amplitude = np.zeros(self.audio_latency, dtype=np.float32)
        self._beat_flag = [False for _ in range(self.audio_latency)]
        self._onset_flag = [False for _ in range(self.audio_latency)]

        # For streaming FFT with overlap
        self.hop_length = self.fft_size // 2
        self.overlap_buffer = np.zeros(self.fft_size - self.hop_length, dtype=np.float32)

    def _set_process_priority(self) -> None:
        """Set process priority for better audio performance."""
        try:
            if os.name == "posix":
                p = psutil.Process(os.getpid())
                p.nice(10)
            elif os.name == "nt":
                thread_id = threading.get_native_id()
                ctypes.windll.kernel32.SetThreadPriority(thread_id, 2)
        except Exception as e:
            warnings.warn(f"Could not set process priority: {e}")

    def set_audio_latency(self, latency: int) -> None:
        """Set audio latency and reinitialize buffers."""
        if latency <= 0:
            raise ValueError("Latency must be positive")
        self.audio_latency = latency
        self._init_analysis_buffers()

    def do_analysis(self, audio_buffer: npt.NDArray[np.float32]) -> None:
        """
        Perform FFT and amplitude analysis on audio buffer.
        Uses streaming FFT with overlap for real-time efficiency.

        Args:
            audio_buffer: Audio samples to analyze
        """
        if not self.analyse:
            return

        try:
            # Calculate amplitude (RMS)
            self.amplitude[self.audio_buffer_write_ptr] = np.sqrt(np.mean(audio_buffer ** 2))

            if self.analyse_fft:
                full_buffer = np.concatenate([self.overlap_buffer, audio_buffer])
                num_frames = (len(full_buffer) - self.fft_size) // self.hop_length + 1
                averaged_magnitude = self.fft(full_buffer, num_frames)
                # Check for onset
                if self.analyse_onsets:
                    if self.check_onset(averaged_magnitude, num_frames):
                        self._onset_flag[0] = True
                        if self.analyse_beats:
                            if hasattr(self, "beat_tracker"):
                                self.beat_tracker.add_onset()

                # Check for beat
                if self.check_beat():
                    self._beat_flag[0] = True

                # ===== END ONSET DETECTION =====

            self.audio_buffer_write_ptr = (self.audio_buffer_write_ptr + 1) % self.audio_latency

        except Exception as e:
            import traceback
            traceback.print_exc()
            warnings.warn(f"Analysis error: {e}")

    def fft(self, full_buffer, num_frames):
        # Calculate how many FFT frames we can extract

        if num_frames > 0:
            fft_accum = np.zeros((self.fft_size // 2) + 1, dtype=np.float32)

            # Process each frame with hop
            for i in range(num_frames):
                frame_start = i * self.hop_length
                frame_end = frame_start + self.fft_size
                frame = full_buffer[frame_start:frame_end]

                # Apply window and FFT
                windowed = frame * self._fft_window
                fft_result = np.fft.rfft(windowed)
                fft_accum += np.abs(fft_result)

            # Average across frames
            averaged_magnitude = fft_accum / num_frames
            self.fft_vals[self.audio_buffer_write_ptr] = averaged_magnitude
            # Update overlap buffer with remaining samples
            consumed = num_frames * self.hop_length
            self.overlap_buffer = full_buffer[consumed:consumed + (self.fft_size - self.hop_length)]
            return averaged_magnitude

    def check_beat(self):
        if hasattr(self, "beat_tracker"):
            return self.beat_tracker.check_beat()

    def check_onset(self, averaged_magnitude, num_frames):
        if hasattr(self, "onset_detector"):
            return self.onset_detector.process_fft_frame(
                averaged_magnitude,
                num_frames=num_frames
            )

    def audio_callback(self) -> npt.NDArray[np.float32]:
        """
        Generate audio samples. Override in subclasses.

        Returns:
            Audio samples
        """
        self.on_new_frame(np.array([]))
        self.internal_callback()
        return np.zeros(self.buffer_size, dtype=np.float32)

    def _get_silence(self) -> npt.NDArray[np.float32]:
        """Return appropriately shaped silence buffer."""
        return np.zeros(self.buffer_size, dtype=np.float32)

    def capture_audio(self) -> None:
        """Main audio capture loop. Runs in separate thread."""
        # Set output device to default if not specified
        if self.output_device is None:
            self.output_device = sd.default.device[1]
            print(f"Output device set to default: {sd.default.device[1]}")

        if self.output_device is not None:
            device_info = sd.query_devices(self.output_device)
            self.channels = device_info['max_output_channels']
            print(f"Channels: {self.channels}")

        print(f"Starting audio: channels={self.channels}, sr={self.sr}, device={self.output_device}")

        try:
            with sd.OutputStream(
                channels=self.channels,
                samplerate=self.sr,
                blocksize=self.buffer_size,
                device=self.output_device
            ) as stream:
                self.stream = stream

                while self.running and not self._shutdown_event.is_set():
                    try:
                        if self.pause_event.is_set():
                            time.sleep(0.01)
                            continue

                        # Get audio from callback
                        audio_data = self.audio_callback()

                        # Ensure 2D array
                        if audio_data.ndim == 1:
                            audio_data = audio_data[np.newaxis, :]

                        # Match channel count
                        audio_data = self._match_channels(audio_data)

                        # Write to stream
                        to_write = np.ascontiguousarray(audio_data.T)

                        # Handle recording
                        if self.recording:
                            self._handle_recording(to_write)

                        stream.write(to_write)

                        # Analyze first channel
                        self.do_analysis(audio_data[0, :])

                    except Exception as e:
                        warnings.warn(f"Audio callback error: {e}")
                        continue

        except Exception as e:
            warnings.warn(f"Stream error: {e}")
        finally:
            self.running = False

    def _match_channels(self, audio_data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Match audio data to required channel count."""
        current_channels = audio_data.shape[0]

        if current_channels == self.channels:
            return audio_data
        elif current_channels < self.channels:
            # Duplicate first channel to fill
            return np.tile(audio_data[0, :], (self.channels, 1))
        else:
            # Trim to required channels
            return audio_data[:self.channels, :]

    def _handle_recording(self, audio_data: npt.NDArray[np.float32]) -> None:
        """Handle recording buffer management."""
        elapsed = time.time() - self._recording_start_time
        if elapsed > AudioConfig.MAX_RECORDING_DURATION:
            self.stop_recording()
            print("Maximum recording duration reached")
        else:
            self.recording_buffer.append(audio_data.copy())

    def start_recording(self) -> None:
        """Start recording audio."""
        self.recording = True
        self.recording_buffer = []
        self._recording_start_time = time.time()
        print("Recording started")

    def stop_recording(self) -> npt.NDArray[np.float32]:
        """
        Stop recording and return recorded audio.

        Returns:
            Recorded audio as numpy array
        """
        self.recording = False
        if self.recording_buffer:
            return np.concatenate(self.recording_buffer, axis=0)
        return np.array([])

    def play(self) -> None:
        """Start audio playback."""
        if not self.running:
            self.running = True
            self._shutdown_event.clear()
            self.pause_event.clear()
            self.play_thread = threading.Thread(target=self.capture_audio, daemon=True)
            self.play_thread.start()

    def pause(self) -> None:
        """Pause audio playback."""
        if self.running:
            self.pause_event.set()

    def resume(self) -> None:
        """Resume audio playback."""
        if self.running and self.pause_event.is_set():
            self.pause_event.clear()

    def stop(self) -> None:
        """Stop audio playback and cleanup."""
        if self.running:
            self.running = False
            self._shutdown_event.set()

            if self.play_thread and self.play_thread.is_alive():
                self.play_thread.join(timeout=AudioConfig.THREAD_JOIN_TIMEOUT)

        try:
            if hasattr(self, 'stream') and self.stream:
                if self.stream.active:
                    self.stream.abort()
                self.stream.close()
                self.stream = None
        except Exception as e:
            warnings.warn(f"Error stopping stream: {e}")
