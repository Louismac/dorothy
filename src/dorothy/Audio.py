import librosa
import numpy as np
import sounddevice as sd
import threading
import librosa
import numpy as np
import os
import psutil
import ctypes
import time

#For Rave example, not normally needed
try:
    import torch
    torch.set_grad_enabled(False)
    from .utils.magnet import preprocess_data, RNNModel, generate
except ImportError:
    print("torch not available, machine learning examples won't work, otherwise ignore.")
#Main class for music analysis and generation

class Audio:
    
    audio_outputs = []  
    clocks = []

    def __init__(self):
        print("Loading Audio Engine")
        print(sd.query_devices())

    def start_magnet_stream(self, model_path, dataset_path, buffer_size=2048, sr = 44100, output_device=None):
        """
        Start stream generating from a pretrained RAVE model
        
        Args:
            model_path (str): The path to the pretrained model
            dataset_path (str): The path to seed audio file used to train model
            buffer_size (int): Size of buffer when playing back / analysing audio. 
            sr (int): Sample rate to capture at
            output_device (int): Where to play this back to. print(sd.query_devices()) to see available
        Returns:
            int: the index of the device in the dot.music.audio_outputs list
        """
        device = MAGNetPlayer(model_path, dataset_path,
                            buffer_size=buffer_size, 
                            sr=sr,output_device = output_device)
        self.audio_outputs.append(device)
        index = len(self.audio_outputs)-1
        self.play(index)
        return index

    def start_rave_stream(self, model_path="",fft_size=1024, buffer_size=2048, sr = 44100, latent_dim=16, output_device=None):
        """
        Start stream generating from a pretrained RAVE model
        
        Args:
            model_path (str): The path to the pretrained model
            fft_size (int): Size of fft
            buffer_size (int): Size of buffer when playing back / analysing audio. 
            sr (int): Sample rate to capture at
            latent_dim (int): The numer of latent dimensions. Must match pretrained model.
            output_device (int): Where to play this back to. print(sd.query_devices()) to see available
        Returns:
            int: the index of the device in the dot.music.audio_outputs list
        """
        device = RAVEPlayer(model_path=model_path, 
                                             buffer_size=buffer_size, 
                                            sr=sr, fft_size=fft_size, 
                                            latent_dim=latent_dim,
                                            output_device = output_device)
        self.audio_outputs.append(device)
        index = len(self.audio_outputs)-1
        self.play(index)
        return index
    
    def update_rave_from_stream(self, input=0):
        """
        Start using a given stream (e.g. a file player or mic input) as input to a RAVE stream
        
        Args:
            input (int): The index in dot.music.audio_outputs to use
        """
        input_device = self.audio_outputs[input]
        def internal_callback():
            with torch.no_grad():
                input_audio = torch.Tensor(input_device.audio_buffer).reshape(1,1,-1)
                for a in self.audio_outputs:
                    if isinstance(a, RAVEPlayer):
                        a.current_latent = a.model.encode(input_audio)
        input_device.gain = 0
        input_device.analyse = False
        input_device.internal_callback = internal_callback

    def start_device_stream(self, device, fft_size=1024, buffer_size=2048, sr = 44100, analyse=True):
        """
        Start stream capturing audio from an input
        
        Args:
            file_path (str): The path to the audio
            device (int): Where to capture audio from (e.g. a mic input). print(sd.query_devices()) to see available
            fft_size (int): Size of fft
            buffer_size (int): Size of buffer when playing back / analysing audio. 
            sr (int): Sample rate to capture at
            analyse (bool): Should this stream be analysed for amplitude, fft etc...
        Returns:
            int: the index of the device in the dot.music.audio_outputs list
        """
        self.audio_outputs.append(AudioCapture(analyse=analyse,
                                          buffer_size=buffer_size, sr=sr, fft_size=fft_size, input_device=device))
        index = len(self.audio_outputs)-1
        self.play(index)
        return index

    def start_file_stream(self, file_path, fft_size=512, buffer_size=1024, sr = 44100, output_device=None, analyse = True):
        """
        Start stream of a given audio file 
        
        Args:
            file_path (str): The path to the audio
            fft_size (int): Size of fft
            buffer_size (int): Size of buffer when playing back / analysing audio. 
            sr (int): Sample rate of the provided audio
            output_device (int): Where to play this back to. print(sd.query_devices()) to see available
            analyse (bool): Should this stream be analysed for amplitude, fft etc...
        Returns:
            int: the index of the device in the dot.music.audio_outputs list
        """
        #load file        
        y, sr = librosa.load(file_path, sr=sr, mono=False)
        return self.start_sample_stream(y, fft_size, buffer_size, sr, output_device, analyse)
    
    def start_dsp_stream(self, audio_callback, fft_size=512, buffer_size=1024, sr = 44100, output_device=None, analyse = True):
        """
        Start stream of a given audio file 
        
        Args:
            audio_callback (str): callback generating the audio
            fft_size (int): Size of fft
            buffer_size (int): Size of buffer when playing back / analysing audio. 
            sr (int): Sample rate of the provided audio
            output_device (int): Where to play this back to. print(sd.query_devices()) to see available
            analyse (bool): Should this stream be analysed for amplitude, fft etc...
        Returns:
            int: the index of the device in the dot.music.audio_outputs list
        """
        self.sr = sr
        self.audio_outputs.append(CustomPlayer(audio_callback,frame_size=buffer_size,
                                              analyse=analyse, fft_size = fft_size, buffer_size = buffer_size, 
                                              sr = sr, output_device=output_device))
        index = len(self.audio_outputs)-1
        self.play(index)
        return index
    
    #Start stream of given audio samples (e.g. we can use this to playback things we make in class)
    def start_sample_stream(self, y, fft_size=1024, buffer_size=1024, sr = 44100, output_device=None, analyse = True):
        """
        Start stream of given audio samples (e.g. we can use this to playback things we make in class, or have loaded from files)
        
        Args:
            y (np.array): The audio!
            fft_size (int): Size of fft
            buffer_size (int): Size of buffer when playing back / analysing audio. 
            sr (int): Sample rate of the provided audio
            output_device (int): Where to play this back to. print(sd.query_devices()) to see available
            analyse (bool): Should this stream be analysed for amplitude, fft etc...
        Returns:
            int: the index of the device in the dot.music.audio_outputs list
        """
        self.y = np.array(y,dtype=np.float32)
        self.sr = sr
        #Beat info
        to_track = self.y if self.y.ndim == 1 else self.y[0,:]
        self.tempo, self.beats = librosa.beat.beat_track(y=to_track, sr=self.sr, units='samples')
        self.beat_ptr = 0
        device = SamplePlayer(y = self.y, analyse=analyse,
                            fft_size = fft_size, buffer_size = buffer_size, sr = self.sr, output_device=output_device)
        self.audio_outputs.append(device)
        index = len(self.audio_outputs)-1
        self.play(index)
        return index
    
    def get_clock(self, bpm=120):
        c = Clock()
        c.set_bpm(bpm)
        self.clocks.append(c)
        return self.clocks[len(self.clocks)-1]
    
    #We actually return a previous value to account for audio latency
    def fft(self, output = 0):
        """
        Return current fft (for visualising)
        
        Args:
            output (int): The audio output to check
        Returns:
            np.array: Average fft values
        """
        if output < len(self.audio_outputs):
            o = self.audio_outputs[output]
            return o.fft_vals[(o.audio_buffer_write_ptr+1)%o.audio_latency]
    
    #We actually return a previous value to account for audio latency
    def amplitude(self, output = 0):
        """
        Return current amplitude (for visualising)
        
        Args:
            output (int): The audio output to check
        Returns:
            float: Average amplitude
        """
        if output < len(self.audio_outputs):
            o = self.audio_outputs[output]
            return o.amplitude[(o.audio_buffer_write_ptr+1)%o.audio_latency]

    def play(self, output=0):
        if output < len(self.audio_outputs):
            o = self.audio_outputs[output]
            o.play()

    def stop(self, output=0):
        if output < len(self.audio_outputs):
            o = self.audio_outputs[output]
            o.stop()

    def clean_up(self):
        for o in self.audio_outputs:
            o.stop()  
        for c in self.clocks:
            c.stop()  
    
    def pause(self, output=0):
        if output < len(self.audio_outputs):
            o = self.audio_outputs[output]
            o.pause()

    def resume(self, output=0):
        if output < len(self.audio_outputs):
            o = self.audio_outputs[output]
            o.resume()

    #Has there been a beat since this was last called?
    def is_beat(self, output=0):
        """
        Has there been a beat since this was last called?
        
        Args:
            output (int): The audio output to check
        Returns:
            bool: Has there been a beat since this was last called?
        """
        cs = self.audio_outputs[output].current_sample
        next_beat = self.beats[self.beat_ptr%len(self.beats)]
        # print(next_beat, self.beat_ptr, cs)
        is_beat = False
        #are we past the most recent beat?
        if next_beat < cs:
            is_beat = True
            self.beat_ptr += 1
        return is_beat

#Parent class for audio providers 
class AudioDevice:
    def __init__(self, on_new_frame = lambda n=1:0,
                 analyse=True, fft_size=1024, buffer_size=2048, sr=44100, output_device=None):
        self.running = False
        self.sr = sr
        self.fft_size = fft_size
        self.audio_latency = 5
        self.audio_buffer_write_ptr = 0
        self.buffer_size = buffer_size
        self.set_audio_latency(5)
        self.analyse = analyse
        self.output_device = output_device
        print(os.name)
        self.on_new_frame = on_new_frame
        self.internal_callback = lambda:0
        self.recording_buffer = []
        self.recording = False

        if self.fft_size > self.buffer_size:
            print("warning, fft window is bigger than buffer, numpy will zero pad, which may lead to unexpected results")

        if os.name == "posix":
            p = psutil.Process(os.getpid())
            p.nice(10)
        elif os.name == "nt":
            thread_id = threading.get_native_id()
            ctypes.windll.kernel32.SetThreadPriority(thread_id, 2)
        sd.default.samplerate = self.sr
        self.channels = 1
        print("output_device", output_device)
        #Set to default if no device provided
        
        self.pause_event = threading.Event()
        self.play_thread = threading.Thread(target=self.capture_audio)
        self.gain = 1

    def set_audio_latency(self, l):
        self.audio_latency = l
        self.fft_vals = [np.zeros((self.fft_size//2)+1) for i in range(self.audio_latency)]
        self.amplitude = np.zeros(self.audio_latency)

    def do_analysis(self, audio_buffer):
        if self.analyse:
            #Get amplitude
            self.amplitude[self.audio_buffer_write_ptr] = np.mean(audio_buffer**2)
            num_frames = 1 + (len(audio_buffer) - self.fft_size) // self.fft_size//2
            fft_results = np.zeros((num_frames, self.fft_size), dtype=complex)
            window = np.hanning(self.fft_size)
            for i in range(num_frames):
                frame_start = i * self.fft_size//2
                frame_end = frame_start + self.fft_size
                frame = audio_buffer[frame_start:frame_end]
                windowed_frame = frame * window
                fft_results[i] = np.fft.fft(windowed_frame)

            self.fft_vals[self.audio_buffer_write_ptr] = np.mean(np.abs(fft_results),axis=0)[:(self.fft_size//2)+1]

            self.audio_buffer_write_ptr = (self.audio_buffer_write_ptr + 1) % self.audio_latency

    #stub (overwritten in subclass)
    def audio_callback(self):
        self.on_new_frame()
        self.internal_callback()
        return np.zeros(self.buffer_size) # Fill buffer with silence
        
    def capture_audio(self):
        #Set to default if no device provided
        if self.output_device is None:
            self.output_device = sd.default.device[1]
            print("output_device set to default", sd.default.device[1])

        if self.output_device is not None:
            self.channels = sd.query_devices(self.output_device)['max_output_channels']
            print("channels:", self.channels)
        
        print("play_audio", "channels", self.channels, self.sr, "output_device",self.output_device)
        with sd.OutputStream(channels=self.channels, samplerate=self.sr, blocksize=self.buffer_size, device=self.output_device) as stream:
            while self.running:
                if not self.pause_event.is_set():
                    audio_data = self.audio_callback()
                    if audio_data.ndim == 1:
                        audio_data = audio_data[np.newaxis, :]
                    #duplicate to fill channels (mostly generating mono)
                    dif = self.channels - audio_data.shape[0]
                    if dif > 0:
                        #Tile out one channel
                        audio_data = np.tile(audio_data[0,:], (self.channels, 1))
                    elif dif < 0:
                        audio_data = audio_data[:self.channels, :]
                    to_write = np.ascontiguousarray(audio_data.T)
                    if self.recording:
                        self.recording_buffer.append(to_write)
                    #Flip axis to write to stream
                    stream.write(to_write)
                    self.do_analysis(audio_data[0,:])
                else:
                    time.sleep(0.1)  
        
    def play(self):
        if not self.running:
            self.running = True
            self.play_thread.start()
            self.pause_event.clear()

    def pause(self):
        if self.running:
            self.pause_event.set()

    def resume(self):
        if self.running and self.pause_event.is_set():
            self.pause_event.clear()

    def stop(self):
        if self.running:
            self.running = False
            self.play_thread.join()

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
        super().__init__(**kwargs)
        
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision('high')
        self.device = torch.device('cpu')
        self.frame_size = 4096//2 #This is the RAVE buffer size 
        self.current_sample = 0
        self.latent_dim = latent_dim
        self.current_latent = torch.randn(1, self.latent_dim, 1).to(self.device)
        self.z_bias = torch.zeros(1,latent_dim,1)
        self.model_path = model_path
        self.model = torch.jit.load(model_path).to(self.device)
        self.current_buffer = self.get_frame()
        self.next_buffer = np.zeros(self.frame_size, dtype = np.float32)
        self.generate_thread = threading.Thread(target=self.fill_next_buffer)
        self.generate_thread.start()
        
    def fill_next_buffer(self):
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
        if self.pause_event.is_set():
            print("paused")
            return np.zeros((self.channels, self.buffer_size), dtype = np.float32) # Fill buffer with silence if paused
        else:
            audio_buffer = self.current_buffer[self.current_sample:self.current_sample +self.buffer_size]
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

#Class for analysing audio streams in realtime
#Doesnt actually play any audio, just analyses and reroutes 
class AudioCapture(AudioDevice):
    def __init__(self, input_device=None, **kwargs):
        super().__init__(**kwargs)
        self.audio_buffer = np.zeros(self.buffer_size)
        self.input_device = input_device

    #Doesnt actually return any audio (its already playing elsewhere)!    
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        if self.pause_event.is_set():
            # If paused, skip processing
            return
        else:
            #Window the current audio buffer and get fft 
            self.audio_buffer = indata[:, 0]
            self.internal_callback()
            self.do_analysis(self.audio_buffer)
            self.on_new_frame(self.audio_buffer)
            # print(self.audio_buffer.shape, indata.shape)
            if self.recording:
                audio_data = indata.copy()
                #trim or duplicate to get 2 channels
                channels = 2
                if audio_data.ndim == 1:
                    audio_data = audio_data[:,np.newaxis]
                #duplicate to fill channels (mostly generating mono)
                dif = channels - audio_data.shape[1]
                # print(audio_data.ndim,audio_data.shape,dif)
                if dif > 0:
                    #Tile out one channel
                    audio_data = np.tile(audio_data, (1,channels))
                elif dif < 0:
                    audio_data = audio_data[:,:channels]
                # print(audio_data.ndim,audio_data.shape)
                self.recording_buffer.append(audio_data)
                

    def capture_audio(self):

        self.channels = min(2,sd.query_devices(self.input_device)['max_input_channels'])
        print("channels:", self.channels)
        
        print("capture_audio (AudioCapture)", self.running, self.input_device, self.channels)
        
        with sd.InputStream(callback=self.audio_callback, 
                            channels=self.channels, 
                            blocksize=self.buffer_size, 
                            samplerate=self.sr,
                            device = self.input_device):
            while self.running:
                # Just sleep and let the callback do all the work
                time.sleep(0.1)

#Class for playing back audio files
class SamplePlayer(AudioDevice):

    def __init__(self, y=[0], **kwargs):
        super().__init__(**kwargs)
        self.y = y
        #Make sure at least 2 dimn
        if self.y.ndim == 1:
            self.y = self.y[np.newaxis, :]
        self.current_sample = 0
    
    def audio_callback(self):
        if self.pause_event.is_set():
            return np.zeros(self.buffer_size) # Fill buffer with silence if paused
        else:
            audio_buffer = self.y[:,self.current_sample:self.current_sample + self.buffer_size]
            # print("audio_buffer", audio_buffer.shape, self.current_sample, self.buffer_size)
            self.current_sample += self.buffer_size
            if self.current_sample > len(self.y[0]):
                wrap_ptr = self.current_sample - len(self.y[0])
                wrap_signal = self.y[:,:wrap_ptr]
                audio_buffer = np.concatenate((audio_buffer,wrap_signal), axis=1)
                self.current_sample = wrap_ptr
            self.audio_buffer = audio_buffer
            self.internal_callback()
            self.on_new_frame(audio_buffer)
            return audio_buffer * self.gain

class Sampler:
    def __init__(self, dot):
        self.samples = [np.zeros(1024)]
        self.sample_pos = [-1 for _ in self.samples]
        #Audio Callback function
        def get_frame(size):
            audio = np.zeros(size)
            for i,p in enumerate(self.sample_pos):
                if p >= 0:
                    end = p+size
                    if end >= len(self.samples[i]):
                        remaining = len(self.samples[i])-p
                        end = p + remaining
                        audio[:remaining] += self.samples[i][p:end]
                        self.sample_pos[i] = -1
                    else: 
                        audio += self.samples[i][p:end]
                        self.sample_pos[i] += size
            return audio
        
        dot.music.start_dsp_stream(get_frame, sr = 22050, buffer_size=64)

    def trigger(self, i):
        self.sample_pos[i] = 0

    def load(self, paths):
        self.samples = [librosa.load(p)[0] for p in paths]
        self.sample_pos = [-1 for _ in self.samples]

class Clock:
    def __init__(self):
        #timing
        self.ticks_per_beat = 4
        self.set_bpm(80)
        self.play_thread = threading.Thread(target=self.tick)
        self.playing = False
        self.on_tick = lambda *args: None
    
    def __del__(self):
        print("Clock is being destroyed, cleaning up...")
        self.stop()
 
    def play(self):
        self.tick_ctr = 0
        self.next_tick = self.tick_length
        self.start_time_millis = int(round(time.time() * 1000))
        print(self.start_time_millis,time.time())
        self.playing = True
        self.play_thread.start()

    def stop(self):
        self.playing = False
        self.play_thread.join()

    def tick(self):
        prev = int(round(time.time() * 1000)) - self.start_time_millis
        while self.playing:
            millis = int(round(time.time() * 1000)) - self.start_time_millis
            if millis > self.next_tick:
                self.tick_ctr += 1
                self.next_tick = millis + self.tick_length
                self.on_tick()
            time.sleep(0.001)
            prev = millis

    def set_bpm(self, bpm = 120):
        self.bpm = bpm
        self.tick_length = 60000 / (self.bpm * self.ticks_per_beat)

    def set_tpb(self, ticks_per_beat = 4):
        self.ticks_per_beat = ticks_per_beat
        self.tick_length = 60000 / (self.bpm * self.ticks_per_beat)