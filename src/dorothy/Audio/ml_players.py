"""
Machine-learning based audio players: MAGNetPlayer, RAVEPlayer, ClusterResult.
"""

import os
import json
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import AudioConfig
from .device import AudioDevice

try:
    import torch
    import torch.nn as nn
    import gin
    from rave import RAVE
    torch.set_grad_enabled(False)
    from ..utils.magnet import preprocess_data, RNNModel, generate
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("Some libraries not available. ML features disabled.", ImportWarning)


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


# Generating audio from MAGNet models https://github.com/Louismac/MAGNet
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
        self.frame_size = 1024 * 75

        self.current_buffer = self.get_frame()
        self.next_buffer = np.zeros(self.frame_size, dtype=np.float32)

        self.generate_thread = threading.Thread(target=self.fill_next_buffer)
        self.generate_thread.start()

    def load_model(self, path):
        model = RNNModel(input_size=1025, hidden_size=128, num_layers=2, output_size=1025)
        checkpoint = path
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        return model

    def load_dataset(self, path):
        n_fft = 2048
        hop_length = 512
        win_length = 2048
        sequence_length = 40
        x_frames, _ = preprocess_data(path, n_fft=n_fft,
                                      hop_length=hop_length, win_length=win_length,
                                      sequence_length=sequence_length)
        return x_frames

    def skip(self, index=0):
        if index < len(self.x_frames):
            self.impulse = self.x_frames[index]

    def fill_next_buffer(self):
        self.next_buffer = self.get_frame()
        print("next buffer filled", self.next_buffer.shape)

    def get_frame(self):
        y = 0
        hop_length = 512
        frames_to_get = int(np.ceil(self.frame_size / hop_length)) + 1
        print("requesting new buffer", self.frame_size, frames_to_get)
        with torch.no_grad():
            y, self.impulse = generate(self.model, self.impulse, frames_to_get, self.x_frames)
        return y[:self.frame_size]

    def audio_callback(self):
        if self.pause_event.is_set():
            return np.zero(self.buffer_size, dtype=np.float32)  # Fill buffer with silence if paused
        else:
            start = self.current_sample
            end = self.current_sample + self.buffer_size
            audio_buffer = self.current_buffer[start:end]
            self.current_sample += self.buffer_size
            # Currently dont do proper wrapping for buffer sizes that arent factors of self.frame_size
            if self.current_sample >= self.frame_size:
                self.current_sample = 0
                self.current_buffer = self.next_buffer.copy()
                if not self.generate_thread.is_alive():
                    self.generate_thread = threading.Thread(target=self.fill_next_buffer)
                    self.generate_thread.start()

            self.internal_callback()
            self.on_new_frame(audio_buffer)
            return audio_buffer * self.gain


# Generating audio from RAVE models https://github.com/acids-ircam/RAVE
class RAVEPlayer(AudioDevice):
    def __init__(self, model_path, latent_dim=128, sr=48000, **kwargs):
        """Initialize RAVE player."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for RAVE")
        super().__init__(sr=sr, **kwargs)

        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision('high')
        self.device = torch.device('cpu')
        self.current_sample = 0
        self.latent_dim = latent_dim
        self.cluster_results = {}

        self.current_latent = torch.randn(1, self.latent_dim, 1).to(self.device)
        self.z_bias = torch.zeros(1, latent_dim, 1).to(self.device)

        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # if its a dir, check for .ckpt and .gin
        if os.path.isdir(model_path):
            # THIS IS REALLY IMPORTANT, IF CONV NOT CACHED, STREAMING BAD
            # https://github.com/acids-ircam/cached_conv
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

        # default to this, however, it is updated based on how much audio is input
        # if RAVE is driven by input audio (not just manual z navigation)
        self.generated_size = AudioConfig.RAVE_FRAME_SIZE
        self.overlap_size = 512
        self.hop_size = self.generated_size - self.overlap_size
        if self.overlap_add:
            self.first_frame = True
            self.input_buffer = np.zeros(self.overlap_size, dtype=np.float32)
            self.output_overlap = np.zeros(self.overlap_size, dtype=np.float32)
            self.window = np.hanning(self.generated_size).astype(np.float32)

        self.current_buffer = self.get_frame()
        self.next_buffer = np.zeros(len(self.current_buffer), dtype=np.float32)

        self._start_generator()

    def _initialize_caches(self):
        """Force cache initialization"""
        dummy = torch.zeros(1, self.model.n_channels, 2 ** 14)

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
            # just take mean (drop std)
            if z.shape[1] == self.latent_dim * 2:
                z = z[:, :self.latent_dim, :]
        self.current_latent = z

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
            # How much audio this will be decoded into
            self.generated_size = len(y)
            self.hop_size = self.generated_size - self.overlap_size
            self.window = np.hanning(self.generated_size).astype(np.float32)

        if self.overlap_add:
            y_new = y[:self.generated_size] if len(y) >= self.generated_size else np.pad(y, (0, self.generated_size - len(y)))
            # Apply Hanning window to the ENTIRE chunk
            y_windowed = y_new * self.window
            # ADD the overlap from previous chunk to the beginning
            y_windowed[:self.overlap_size] += self.output_overlap
            # Save the END for next chunk's overlap
            self.output_overlap = y_windowed[-self.overlap_size:].copy()
            # Return the hop_size portion (everything except the overlap at the end)
            y_windowed = y_windowed[:self.hop_size]
            return y_windowed
        else:
            return y

    def audio_callback(self):
        """Generate audio samples."""
        if self.pause_event.is_set():
            return self._get_silence()
        else:
            audio_buffer = self.current_buffer[self.current_sample:self.current_sample + self.buffer_size]
            self.current_sample += self.buffer_size
            # Currently dont do proper wrapping for buffer sizes that arent factors of self.frame_size
            if self.current_sample >= len(self.current_buffer):
                self.current_sample = 0
                self.current_buffer = self.next_buffer.copy()
                if not self.generate_thread.is_alive():
                    # this is where get_frame is called (e.g. current z is decoded and stored in next buffer)
                    # existing next buffer becomes current buffer
                    self._start_generator()

            self.internal_callback()
            self.on_new_frame(audio_buffer)
            return audio_buffer * self.gain

    def toggle_bend(self,
                    function: str,
                    layer_name: str,
                    cluster_id: int = 0):
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
        self, function, layer_name, cluster_id
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

                # Update phase for next call (wrap to 0-2pi)
                state['phase'] = (current_phase + phase_per_sample * time_steps) % (2 * np.pi)

                return output

        handle = layer.register_forward_hook(lfo_hook)
        self.hooks["lfo"][layer_name][cluster_id] = handle

    def load_cluster_results(self, output_dir):
        self.hooks = {"noise": {}, "ablation": {}, "lfo": {}}
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
