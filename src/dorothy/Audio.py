"""
Dorothy Audio Engine - Refactored
A clean, efficient audio processing engine for creative coding.
"""

# Standard library
import os
import time
import threading
import ctypes
import warnings
from typing import Optional, Callable, List, Union
from dataclasses import dataclass

# Third-party
import numpy as np
import numpy.typing as npt
import librosa
import sounddevice as sd
import psutil

# Optional dependencies
try:
    import torch
    import torch.nn as nn
    import gin
    from rave import RAVE
    torch.set_grad_enabled(False)
    from .utils.magnet import preprocess_data, RNNModel, generate
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("Some libraries not available. ML features disabled.", ImportWarning)


# ============================================================================
# Configuration & Constants
# ============================================================================

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


# ============================================================================
# Main Audio Class
# ============================================================================

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
            
    def is_onset(self,output: int = 0) -> bool:
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
        ptr =0  
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


# ============================================================================
# Audio Device Base Class
# ============================================================================

class AudioDevice:
    """Base class for all audio providers."""
    
    def __init__(
        self,
        on_new_frame: Callable[[npt.NDArray[np.float32]], None] = lambda x: None,
        analyse: bool = True,
        fft_size: int = AudioConfig.DEFAULT_FFT_SIZE,
        buffer_size: int = AudioConfig.DEFAULT_BUFFER_SIZE,
        sr: int = AudioConfig.DEFAULT_SAMPLE_RATE,
        output_device: Optional[int] = None
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
        #requires fft
        self.analyse_onsets = False
        #frequires onsets
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

        #if SamplePlayer then using offline detection
        if not isinstance(self,SamplePlayer):
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


# ============================================================================
# Specialized Audio Device Classes
# ============================================================================

#Generating audio from MAGNet models https://github.com/Louismac/MAGNet
class MAGNetPlayer(AudioDevice):
    def __init__(self, model_path, dataset_path, **kwargs):
        super().__init__(**kwargs)
        
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision('high')
        self.device = torch.device('cpu')
        
        self.x_frames = self.load_dataset(dataset_path)
        self.model = self.load_model(model_path)
        
        self.current_sample = 0
        self.impulse = self.x_frames[np.random.randint(self.x_frames.shape[1])]
        self.sequence_length = 40
        self.frame_size = 1024*75

        self.current_buffer = self.get_frame()
        self.next_buffer = np.zeros(self.frame_size, dtype = np.float32)
        
        self.generate_thread = threading.Thread(target=self.fill_next_buffer)
        self.generate_thread.start()
        
    def load_model(self, path):
        model = RNNModel(input_size=1025, hidden_size=128, num_layers=2, output_size=1025)
        checkpoint = path
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        return model

    def load_dataset(self, path):
        n_fft=2048
        hop_length=512
        win_length=2048
        sequence_length = 40
        x_frames, _ = preprocess_data(path, n_fft=n_fft, 
                                            hop_length=hop_length, win_length=win_length, 
                                            sequence_length=sequence_length)
        return x_frames

    def skip(self, index = 0):
        if index < len(self.x_frames):
            self.impulse = self.x_frames[index]

    def fill_next_buffer(self):
        self.next_buffer = self.get_frame()
        print("next buffer filled", self.next_buffer.shape)

    def get_frame(self):
        y = 0
        hop_length=512
        frames_to_get = int(np.ceil(self.frame_size/hop_length))+1
        print("requesting new buffer", self.frame_size, frames_to_get)
        with torch.no_grad():
            y, self.impulse = generate(self.model, self.impulse, frames_to_get, self.x_frames)
        return y[:self.frame_size]

    def audio_callback(self):
        if self.pause_event.is_set():
            return np.zero(self.buffer_size, dtype = np.float32) # Fill buffer with silence if paused
        else:
            start = self.current_sample
            end = self.current_sample + self.buffer_size
            audio_buffer = self.current_buffer[start:end]
            self.current_sample += self.buffer_size
            #Currently dont do proper wrapping for buffer sizes that arent factors of self.frame_size
            if self.current_sample >= self.frame_size:
                self.current_sample = 0
                self.current_buffer = self.next_buffer.copy()
                if not self.generate_thread.is_alive():
                    self.generate_thread = threading.Thread(target=self.fill_next_buffer)
                    self.generate_thread.start()
            
            self.internal_callback()
            self.on_new_frame(audio_buffer)
            return audio_buffer * self.gain


class CustomPlayer(AudioDevice):
    def __init__(self, get_frame, frame_size = 512, **kwargs):
        super().__init__(**kwargs)        
        self.frame_size = frame_size
        self.get_frame = get_frame
        self.current_sample = 0
        self.current_buffer = np.zeros(self.frame_size, dtype = np.float32)
        self.next_buffer = np.zeros(self.frame_size, dtype = np.float32)
        self.generate_thread = threading.Thread(target=self.fill_next_buffer)
        self.generate_thread.start()
        
    def fill_next_buffer(self):
        if self.get_frame is not None:
            self.next_buffer = self.get_frame(self.frame_size).astype(np.float32)

    def audio_callback(self):
        if self.pause_event.is_set():
            print("paused")
            return np.zeros((self.channels, self.buffer_size), dtype = np.float32) # Fill buffer with silence if paused
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



from pathlib import Path
import json


@dataclass
class ClusterResult:
    """Store clustering results"""
    cluster_labels: np.ndarray
    n_clusters: int
    silhouette_score: float
    cluster_centers: Optional[np.ndarray] = None
    method: Optional[str] = None
    pca_components: Optional[np.ndarray] = None
    explained_variance: Optional[np.ndarray] = None


#Generating audio from RAVE models https://github.com/acids-ircam/RAVE
class RAVEPlayer(AudioDevice):
    def __init__(self, model_path, latent_dim=128, sr = 48000, **kwargs):
        """Initialize RAVE player."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for RAVE")
        super().__init__(sr = sr, **kwargs)
        
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision('high')
        self.device = torch.device('cpu')
        self.current_sample = 0
        self.latent_dim = latent_dim
        self.cluster_results = {}

        self.current_latent = torch.randn(1, self.latent_dim, 1).to(self.device)
        self.z_bias = torch.zeros(1,latent_dim,1).to(self.device)

        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        #if its a dir, check for .ckpt and .gin
        if os.path.isdir(model_path):
            #THIS IS REALLY IMPORTANT, IF CONV NOT CACHED, STREAMING BAD
            #https://github.com/acids-ircam/cached_conv
            import cached_conv as cc
            cc.use_cached_conv(True)

            model_path = Path(model_path)
            ckpt_file = list(model_path.glob('*.ckpt'))[0]
            gin_file = list(model_path.glob('*.gin'))[0]
            gin.parse_config_file(gin_file)
            self.model = RAVE.load_from_checkpoint(
                ckpt_file,
                strict=False
            ).eval().to(self.device)
            self._initialize_caches()
            self.overlap_add = False
        else:
            self.model = torch.jit.load(model_path).to(self.device)
            self.overlap_add = False

        #default to this, however, it is updated based on how much audio is input 
        #if RAVE is driven by input audio (not just manual z navigation)
        self.generated_size = AudioConfig.RAVE_FRAME_SIZE
        self.overlap_size = 512
        self.hop_size = self.generated_size - self.overlap_size
        if self.overlap_add:
            self.first_frame = True
            self.input_buffer = np.zeros(self.overlap_size, dtype=np.float32)
            self.output_overlap = np.zeros(self.overlap_size, dtype=np.float32)
            self.window = np.hanning(self.generated_size).astype(np.float32)

        self.current_buffer = self.get_frame()
        self.next_buffer = np.zeros(len(self.current_buffer), dtype = np.float32)
        
        self._start_generator()

    def _initialize_caches(self):
        """Force cache initialization"""
        dummy = torch.zeros(1, self.model.n_channels, 2**14)
        
        with torch.no_grad():
            # This creates cache buffers
            _ = self.model.encode(dummy)
        
        # Prevent cache reset by monkey-patching
        for module in self.model.modules():
            if hasattr(module, 'cache') and module.cache is not None:
                # Store original forward
                original_forward = module.forward
                
                def persistent_forward(self, x, _original=original_forward):
                    # Skip cache initialization check
                    # (cache is already initialized)
                    return _original(x)
                
                # Replace forward to prevent reset
                module.forward = persistent_forward.__get__(module, module.__class__)

    def _start_generator(self) -> None:
        """Start background generation thread."""
        self.generate_thread = threading.Thread(target=self._fill_next_buffer, daemon=True)
        self.generate_thread.start()
        
    def _fill_next_buffer(self):
        self.next_buffer = self.get_frame()
        # print(f"_fill_next_buffer {self.next_buffer.shape}")

    def update_z(self, input_audio):

        if self.overlap_add:
            # Concatenate with previous input for context
            full_input = np.concatenate([self.input_buffer, input_audio])
        else:
            full_input = input_audio
        
        # Encode
        with torch.no_grad():
            x = torch.from_numpy(full_input).reshape(1, 1, -1).to(self.device)
            x = x.repeat(1, self.model.n_channels, 1)
            z = self.model.encode(x)
            #just take mean (drop std)
            if z.shape[1] == self.latent_dim*2:
                z = z[:, :self.latent_dim, :]
        self.current_latent = z
        
        # print(f"encoding{full_input.shape} audio, making {self.current_latent.shape[2]} latents")
        if self.overlap_add:
            # Save the END of the new input for next time
            self.input_buffer = input_audio[-self.overlap_size:].copy()
    
    def get_frame(self):
        
        if self.current_latent is None:
            return np.zeros(self.generated_size, dtype=np.float32)
        
        with torch.no_grad():
            # Decode
            y = self.model.decode(self.current_latent)
            y = y.cpu().numpy().squeeze()
            #How much audio this will be decoded into
            self.generated_size = len(y)
            # print(f"generated {self.generated_size} audio")
            self.hop_size = self.generated_size - self.overlap_size
            self.window = np.hanning(self.generated_size).astype(np.float32)
        
        if self.overlap_add:
            y_new = y[:self.generated_size] if len(y) >= self.generated_size else np.pad(y, (0, self.generated_size - len(y)))
            #Apply Hanning window to the ENTIRE chunk
            y_windowed = y_new * self.window
            # ADD the overlap from previous chunk to the beginning
            y_windowed[:self.overlap_size] += self.output_overlap
            # Save the END for next chunk's overlap
            self.output_overlap = y_windowed[-self.overlap_size:].copy()  
            # Return the hop_size portion (everything except the overlap at the end) 
            y_windowed = y_windowed[:self.hop_size]
            # print(f"returning {len(y_windowed)} audio")
            return y_windowed
        else:
            return y

    def audio_callback(self):
        """Generate audio samples."""
        if self.pause_event.is_set():
            return self._get_silence()
        else:
            audio_buffer = self.current_buffer[self.current_sample:self.current_sample +self.buffer_size]
            self.current_sample += self.buffer_size
            #Currently dont do proper wrapping for buffer sizes that arent factors of self.frame_size
            if self.current_sample >= len(self.current_buffer):
                # print(f"reached end of buffer {self.current_sample} / {len(self.current_buffer)}")
                self.current_sample = 0
                self.current_buffer = self.next_buffer.copy()
                if not self.generate_thread.is_alive():
                    #this is where get_frame is called (e.g. current z is decoded and stored in next buffer)
                    #existing next buffer becomes current buffer
                    self._start_generator()

            self.internal_callback()
            self.on_new_frame(audio_buffer)
            return audio_buffer * self.gain
        
    def toggle_bend(self,
        function:str,
        layer_name: str,
        cluster_id: int=0):
        if self.hooks[function][layer_name][cluster_id] is not None:
            self.hooks[function][layer_name][cluster_id].remove()
            self.hooks[function][layer_name][cluster_id] = None
            print(f"remove {function} for cluster {cluster_id} in layer {layer_name}")
            return False
        else:
            if function == "ablation":
                self.add_ablate_hook(layer_name, cluster_id)
            elif function == "noise":
                self.add_noise_hook(layer_name, cluster_id)
            elif function == "lfo":
                self.add_lfo_hook(layer_name, cluster_id)
            print(f"adding {function} for cluster {cluster_id} in layer {layer_name}")
            return True

    def add_ablate_hook(
        self,
        layer_name: str,
        cluster_id: int = 0,
        ablation_value: float = 0.0
    ):
        """
        Ablate (zero out or set to constant) specific neurons
        
        Args:
            layer_name: Name of layer
            cluster_neurons: Indices of neurons to ablate
            ablation_value: Value to set neurons to (default 0)
        """
        layer = self._get_layer(layer_name)
        layer_clusters = self.cluster_results[layer_name]
        neuron_indices = np.where(np.array(layer_clusters.cluster_labels) == cluster_id)[0].tolist()

        def ablation_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                
                if len(output.shape) == 3:  # (batch, channels, time)
                    output[:, neuron_indices, :] = ablation_value
            
            return output
        handle = layer.register_forward_hook(ablation_hook)
        self.hooks["ablation"][layer_name][cluster_id] = handle
    
    def remove_hook(
        self,function,layer_name,cluster_id        
    ): 
        hook = self.hooks[function][layer_name][cluster_id]
        if hook is not None:
            hook.remove()
            self.hooks[function][layer_name][cluster_id] = None

    def add_noise_hook(
        self,
        layer_name: str,
        cluster_id: int = 0,
        noise_std: float = 0.1,
        noise_type: str = 'gaussian'
    ):
        """
        Add noise to neuron activations
        
        Args:
            layer_name: Layer name
            cluster_neurons: Neurons to add noise to
            noise_std: Standard deviation of noise
            noise_type: 'gaussian' or 'uniform'
        """
        layer = self._get_layer(layer_name)
        layer_clusters = self.cluster_results[layer_name]
        neuron_indices = np.where(np.array(layer_clusters.cluster_labels) == cluster_id)[0].tolist()
        
        def noise_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                noise_shape = list(output.shape)
                
                if noise_type == 'gaussian':
                    noise = torch.randn(noise_shape, device=output.device) * noise_std
                elif noise_type == 'uniform':
                    noise = (torch.rand(noise_shape, device=output.device) - 0.5) * 2 * noise_std
                else:
                    raise ValueError(f"Unknown noise type: {noise_type}")
                
                # Apply noise to cluster neurons only
                if len(output.shape) == 3:  # (batch, channels, time)
                    output[:, neuron_indices, :] += noise[:, neuron_indices, :]
            
            return output
        
        handle = layer.register_forward_hook(noise_hook)
        self.hooks["noise"][layer_name][cluster_id] = handle


    def add_lfo_hook(
        self,
        layer_name: str,
        cluster_id: int = 0,
        freq: float = 10,  # Hz
        amp: float = 1.0
    ):
        """
        Add LFO modulation to cluster neuron activations
        
        Args:
            layer_name: Layer name
            cluster_id: Which cluster to modulate
            freq: LFO frequency in Hz
            amp: Modulation depth (0-1 range recommended)
            phase_offset: Starting phase in radians
        """
        layer = self._get_layer(layer_name)
        layer_clusters = self.cluster_results[layer_name]
        neuron_indices = np.where(np.array(layer_clusters.cluster_labels) == cluster_id)[0].tolist()
        if not hasattr(self, 'latent_sample_rate'):
            test_audio = torch.randn(1, 1, 48000)  # Should be 1 second
            print(f"Audio duration: {test_audio.shape[-1] / 48000:.2f} seconds")

            test_z = self.model.encode(test_audio)[:, :128, :]
            print(f"Latent timesteps: {test_z.shape[-1]}")
            print(f"Expected latent rate: {test_z.shape[-1] / 1.0:.1f} Hz")
            self.latent_sample_rate = test_z.shape[-1] / 1.0
        
        # Initialize LFO state dict if needed
        if not hasattr(self, 'lfo_states'):
            self.lfo_states = {}
        
        # Create unique key for this layer+cluster
        lfo_key = f"{layer_name}_cluster{cluster_id}"
        self.lfo_states[lfo_key] = {
            'phase': 0,  
            'freq': freq,
            'amp': amp
        }
        
        def lfo_hook(module, input, output):
            if isinstance(output, torch.Tensor) and len(output.shape) == 3:
                batch, channels, time_steps = output.shape
                
                state = self.lfo_states[lfo_key]
                current_phase = state['phase']
                
                # Generate time vector
                t = torch.arange(time_steps, device=output.device).float()
                
                # Calculate phase progression for this chunk
                phase_per_sample = 2 * np.pi * state['freq'] / self.latent_sample_rate
                
                # Generate oscillation starting from current_phase
                oscillation = state['amp'] * torch.sin(current_phase + phase_per_sample * t)
                oscillation = oscillation.view(1, 1, -1)
                
                # Modulate cluster neurons
                output[:, neuron_indices, :] *= (1 + oscillation)
                
                # Update phase for next call (wrap to 0-2π)
                state['phase'] = (current_phase + phase_per_sample * time_steps) % (2 * np.pi)
                
                return output
        
        handle = layer.register_forward_hook(lfo_hook)
        self.hooks["lfo"][layer_name][cluster_id] = handle
        
    def load_cluster_results(self, output_dir):
        self.hooks = {"noise":{},"ablation":{},"lfo":{}}            
        # Check if we have any cluster results
        if not self.cluster_results:
            results_path = Path(output_dir) / "clustering_results.json"
            with open(results_path, 'r') as f:
                from_file = json.load(f)
                for layer_name in from_file.keys():
                    res_dict = from_file[layer_name]
                    result = ClusterResult(
                        silhouette_score=res_dict["silhouette_score"],
                        cluster_labels=res_dict["cluster_labels"],
                        n_clusters=res_dict["n_clusters"],
                        explained_variance=res_dict["explained_variance"]
                    )
                    self.hooks["noise"][layer_name] = [None for _ in range(res_dict["n_clusters"])]
                    self.hooks["ablation"][layer_name] = [None for _ in range(res_dict["n_clusters"])]
                    self.hooks["lfo"][layer_name] = [None for _ in range(res_dict["n_clusters"])]
                    self.cluster_results[layer_name] = result
    
    def _get_layer(self, layer_name: str):
        """Get layer module by name"""
        decoder = self.model.decoder
        for name, module in decoder.named_modules():
            if name == layer_name:
                return module

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
        # print("handle recording", elapsed, self._recording_start_time)
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


class SamplePlayer(AudioDevice):
    """Play back audio samples"""
    
    def __init__(
        self,
        y: npt.NDArray[np.float32],
        **kwargs
    ):
        """Initialize sample player."""
        super().__init__(**kwargs)
        
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
                sr = self.sr, 
                units = "samples", delta = 0.2) 
            print(f"found {len(self.beats)} beats and {len(self.onsets)} onsets")  
        self.current_sample = 0
        self.beat_ptr = 0
        self.onset_ptr = 0

    def check_onset(self, *args, **kwargs) -> bool:
        """Check if a onset has occurred since last call."""
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


# ============================================================================
# Sampler Class
# ============================================================================

class Sampler:
    """Simple sampler for triggering audio samples."""
    
    def __init__(self, audio_instance: Audio):
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


# ============================================================================
# Clock Class
# ============================================================================

class Clock:
    """Simple clock for timing-based operations."""
    
    def __init__(self):
        """Initialize clock."""
        self.ticks_per_beat = 4
        self.bpm = 120
        self.tick_length = 0.0
        self.tick_ctr = 0
        self.next_tick = 0.0
        self.start_time_millis = 0
        self.playing = False
        self._shutdown_event = threading.Event()
        self.play_thread: Optional[threading.Thread] = None
        
        # Callback
        self.on_tick: Callable[[], None] = lambda: None
        
        # Initialize timing
        self.set_bpm(120)
 
    def play(self) -> None:
        """Start the clock."""
        if self.playing:
            return
        
        self.tick_ctr = 0
        self.start_time_millis = int(round(time.time() * 1000))
        self.next_tick = self.tick_length
        self.playing = True
        self._shutdown_event.clear()
        
        self.play_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self.play_thread.start()

    def stop(self) -> None:
        """Stop the clock."""
        if not self.playing:
            return
        
        self.playing = False
        self._shutdown_event.set()
        
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=AudioConfig.THREAD_JOIN_TIMEOUT)

    def _tick_loop(self) -> None:
        """Main tick loop running in separate thread."""
        while self.playing and not self._shutdown_event.is_set():
            try:
                millis = int(round(time.time() * 1000)) - self.start_time_millis
                
                if millis >= self.next_tick:
                    self.tick_ctr += 1
                    self.next_tick = millis + self.tick_length
                    
                    try:
                        self.on_tick()
                    except Exception as e:
                        warnings.warn(f"Clock callback error: {e}")
                
                time.sleep(0.001)
            except Exception as e:
                warnings.warn(f"Clock tick error: {e}")

    def set_bpm(self, bpm: float = 120.0) -> None:
        """
        Set beats per minute.
        
        Args:
            bpm: Beats per minute
        """
        if bpm <= 0:
            raise ValueError("BPM must be positive")
        
        self.bpm = bpm
        self.tick_length = 60000.0 / (self.bpm * self.ticks_per_beat)

    def set_tpb(self, ticks_per_beat: int = 4) -> None:
        """
        Set ticks per beat.
        
        Args:
            ticks_per_beat: Number of ticks per beat
        """
        if ticks_per_beat <= 0:
            raise ValueError("Ticks per beat must be positive")
        
        self.ticks_per_beat = ticks_per_beat
        self.tick_length = 60000.0 / (self.bpm * self.ticks_per_beat)

from collections import deque

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
        self.pre_max =  int(0.03 * self.sample_rate // self.hop_length)  # 30ms
        self.post_max = int(0.00 * self.sample_rate // self.hop_length + 1) # 0ms
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
        if onset_detected:
            print("onset")
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
                ioi = onset_list[i] - onset_list[i-1]
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
                print(f"🔒 TEMPO LOCKED: {self.locked_tempo_bpm:.1f} BPM (std: {tempo_std:.2f})")
        
        # NEW: Unlock if tempo has been consistently wrong
        if self.tempo_locked and self.beats_at_locked_tempo > 20:
            # Check if locked tempo is still valid
            recent_tempo_std = np.std(list(self.tempo_history)[-5:])
            if recent_tempo_std > 8.0:  # Tempo is drifting
                print(f"🔓 TEMPO UNLOCKED (drift detected: {recent_tempo_std:.2f})")
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
        
        # Debug output
        # print(f"predicted phase:{best_phase_offset}")
    
    def check_beat(self) -> bool:
        current_time_seconds = time.time()-self.start_time
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
    
    def get_tempo_distribution(self) -> tuple[np.ndarray, np.ndarray]:
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