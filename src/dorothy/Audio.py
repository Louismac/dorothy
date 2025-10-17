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
    torch.set_grad_enabled(False)
    from .utils.magnet import preprocess_data, RNNModel, generate
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. ML features disabled.", ImportWarning)


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
                input_audio = torch.Tensor(input_device.audio_buffer).reshape(1, 1, -1)
                for device in self.audio_outputs:
                    if isinstance(device, RAVEPlayer):
                        device.current_latent = device.model.encode(input_audio)
        
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
        tempo, beats = librosa.beat.beat_track(y=to_track, sr=sr, units='samples')
        
        device = SamplePlayer(
            y=y,
            tempo=tempo,
            beats=beats,
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
        if not isinstance(device, SamplePlayer):
            return False
            
        return device.check_beat()

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
            
            # Streaming FFT with overlap-add
            # Concatenate overlap buffer with new audio
            full_buffer = np.concatenate([self.overlap_buffer, audio_buffer])
            
            # Calculate how many FFT frames we can extract
            num_frames = (len(full_buffer) - self.fft_size) // self.hop_length + 1
            
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
                self.fft_vals[self.audio_buffer_write_ptr] = fft_accum / num_frames
                
                # Update overlap buffer with remaining samples
                consumed = num_frames * self.hop_length
                self.overlap_buffer = full_buffer[consumed:consumed + (self.fft_size - self.hop_length)]
            
            self.audio_buffer_write_ptr = (self.audio_buffer_write_ptr + 1) % self.audio_latency
            
        except Exception as e:
            warnings.warn(f"Analysis error: {e}")

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


#Generating audio from RAVE models https://github.com/acids-ircam/RAVE
class RAVEPlayer(AudioDevice):
    def __init__(self, model_path, latent_dim=128, **kwargs):
        """Initialize RAVE player."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for RAVE")
        
        super().__init__(**kwargs)
        
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision('high')
        self.device = torch.device('cpu')
        self.frame_size = AudioConfig.RAVE_FRAME_SIZE #This is the RAVE buffer size 
        self.current_sample = 0
        self.latent_dim = latent_dim

        self.current_latent = torch.randn(1, self.latent_dim, 1).to(self.device)
        self.z_bias = torch.zeros(1,latent_dim,1)

        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = torch.jit.load(model_path).to(self.device)

        self.current_buffer = self.get_frame()
        self.next_buffer = np.zeros(self.frame_size, dtype = np.float32)
        
        # Start generation thread
        self._start_generator()

    def _start_generator(self) -> None:
        """Start background generation thread."""
        self.generate_thread = threading.Thread(target=self._fill_next_buffer, daemon=True)
        self.generate_thread.start()
        
    def _fill_next_buffer(self):
        self.next_buffer = self.get_frame()

    def get_frame(self):
        y = 0
        with torch.no_grad():
            z = self.current_latent
            y = self.model.decode(z + self.z_bias)
            y = y.reshape(-1).to(self.device).numpy()
        #Drop second half (RAVE gives us stereo end to end)
        return y[:self.frame_size]

    def audio_callback(self):
        """Generate audio samples."""
        if self.pause_event.is_set():
            return self._get_silence()
        else:
            audio_buffer = self.current_buffer[self.current_sample:self.current_sample +self.buffer_size]
            self.current_sample += self.buffer_size

            #Currently dont do proper wrapping for buffer sizes that arent factors of self.frame_size
            if self.current_sample >= self.frame_size:
                self.current_sample = 0
                self.current_buffer = self.next_buffer.copy()
                if not self.generate_thread.is_alive():
                    self._start_generator()

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
    """Play back audio samples with beat tracking."""
    
    def __init__(
        self,
        y: npt.NDArray[np.float32],
        tempo: Optional[float] = None,
        beats: Optional[npt.NDArray[np.int32]] = None,
        **kwargs
    ):
        """Initialize sample player."""
        super().__init__(**kwargs)
        
        # Ensure audio is 2D
        self.y = y if y.ndim == 2 else y[np.newaxis, :]
        self.current_sample = 0
        
        # Beat tracking
        self.tempo = tempo
        self.beats = beats if beats is not None else np.array([])
        self.beat_ptr = 0
    
    def check_beat(self) -> bool:
        """Check if a beat has occurred since last call."""
        if len(self.beats) == 0:
            return False
        
        next_beat = self.beats[self.beat_ptr % len(self.beats)]
        
        if next_beat < self.current_sample:
            self.beat_ptr += 1
            return True
        
        return False
    
    def audio_callback(self) -> npt.NDArray[np.float32]:
        """Generate audio samples from stored audio."""
        if self.pause_event.is_set():
            return self._get_silence()
        
        # Get audio buffer
        audio_buffer = self.y[:, self.current_sample:self.current_sample + self.buffer_size]
        
        # Advance playhead
        self.current_sample += self.buffer_size
        
        # Handle wrapping
        if self.current_sample >= self.y.shape[1]:
            wrap_ptr = self.current_sample - self.y.shape[1]
            wrap_signal = self.y[:, :wrap_ptr]
            audio_buffer = np.concatenate((audio_buffer, wrap_signal), axis=1)
            self.current_sample = wrap_ptr
        
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