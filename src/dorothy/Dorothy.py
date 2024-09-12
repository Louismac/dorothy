import numpy as np
import cv2
from cv2 import rectangle, line
import signal
import sys
import sounddevice as sd
import threading
import librosa
import numpy as np
import os
import psutil
import ctypes
import time
import traceback
import glob
from .css_colours import css_colours

#For Rave example, not normally needed
try:
    import torch
    torch.set_grad_enabled(False)
    from .utils.magnet import preprocess_data, RNNModel, generate
except ImportError:
    print("torch not available, machine learning examples won't work, otherwise ignore.")

class Dorothy:

    """
    Main drawing class
    """
    
    width = 640
    height = 480
    frame = 0
    mouse_x = 1
    mouse_y = 1
    mouse_down = False
    start_time_millis = int(round(time.time() * 1000))
    millis = 0
    layers = []
    on_key_pressed = lambda *args: None
    recording = False
    recording_buffer = []
    test = 0

    def __init__(self, width = 640, height = 480):

        """
        Initialize the class with a value.
        
        Args:
            width (int): The width of the canvas. Defaults to 640
            height (int): The height of the canvas. Defaults to 480
        """

        print("Loading Main Library")   

        self.width = width
        self.height = height
        self.canvas = np.ones((height,width,3), np.uint8)*255
        self.load_colours()
        self.music = Audio()
    
    def load_colours(self):
        for colour_name, rgb in css_colours.items():
            colour_name = colour_name.replace(" ", "_")
            setattr(self, colour_name, rgb)

   
    def get_layer(self):
        """
        Returns a new layer for drawing
        
        Returns:
            np.array: of ones h x w x channels (3)
        """
        return np.ones((self.height,self.width,3), np.uint8)
    
    #Push layer back onto stack
    def draw_layer(self, c, alpha=1):
        """
        Adds layer to render stack
        
        Args:
            c (np.array): The layer to add
            alpha (float): Transparency value between 0 and 1. Defaults to 1.
        """
        self.layers.append([c, alpha])
    
    #Perform a linear transformation given matrix a
    def linear_transformation(self, src, a, origin =(0,0)):
        
        rows, cols = src.shape[:2]
        
        translate_to_origin = np.array([[1, 0, -origin[0]],
                                        [0, 1, -origin[1]],
                                        [0, 0, 1]])
        
        translate_back = np.array([[1, 0, origin[0]],
                                [0, 1, origin[1]],
                                [0, 0, 1]])
        
        transformation_matrix = np.array([[a[0,0], a[0,1], 0],
                                        [a[1,0],a[1,1], 0],
                                        [0, 0, 1]])
        
        transformation_matrix = translate_back @ transformation_matrix @ translate_to_origin

        transformed_image = cv2.warpAffine(src, transformation_matrix[:2, :], (cols, rows))

        return transformed_image

    
    def transform(self, canvas, m, origin = (0,0)):
        """
        Transform a canvas given the linear matrix (2,2) and an origin   

        Args:
            canvas (np.array): The layer to scale
            m (np.array): A 2x2 matrix for the linear transform
            origin (tuple): The origin about which to make the transform

        Returns:
            np.array: The transformed version of the layer
        """
        return self.linear_transformation(canvas, m, origin)
    
    def scale(self, canvas, sx=1, sy=1, origin =(0,0)):
        """
        Scale canvas given x and y factors and an origin   

        Args:
            canvas (np.array): The layer to scale
            sx (float): The scale factor in the x axis. Defaults to 1
            sy (float): The scale factor in the y axis. Defaults to 1
            origin (tuple): The origin about which to make the transform

        Returns:
            np.array: The transformed version of the layer
        """
        m = np.array([[sx,0.0],
                          [0.0,sy]])
        return self.transform(canvas, m, origin)
    
    def rotate(self, canvas, theta, origin = (0,0)):
        """
        Rotate canvas given theta and an origin 

        Args:
            canvas (np.array): The layer to scale
            theta (float): The rotation in radians
            origin (tuple): The origin about which to make the transform

        Returns:
            np.array: The transformed version of the layer
        """

        m = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
        return self.transform(canvas, m, origin)
    
    #Callback for mouse moved
    def mouse_moved(self, event, x, y, flags, param):
        self.mouse_x = x
        self.mouse_y = y
        if event == 1:
            self.mouse_down = True
        elif event == 4:
            self.mouse_down = False

    def background(self, col=(0,0,0)):
        """
        Fill canvas with given RGB colour

        Args:
            col (tuple): The RGB colour (unit8) to fill. Defaults to black.
        """
        rectangle(self.canvas, (0,0), (self.width,self.height), col, -1)

    def draw_waveform(self, layer, audio_output = 0, col=(0,0,0), with_playhead = False):
        """
        Draw current waveform loaded into given audio output to layer

        Args:
            layer (np.array): The layer to draw to
            audio_output (int): The index of the audio device to show
            col (tuple): RGB Colour to draw the waveform. Defaults to black.
            with_playhead (bool): Show current position in audio. Defaults to false.
        Returns:
            layer (np.array): The layer that has been updated 
        """
        if audio_output < len(self.music.audio_outputs):
            output = self.music.audio_outputs[audio_output]
            if isinstance(output, SamplePlayer):
                playhead = int((output.current_sample / len(output.y)) *self.width)
                step = len(output.y) // self.width
                for i, val in enumerate(output.y[::step]):
                    h = int(val * self.height)
                    y = self.height//2 - h//2
                    line(layer, (i, y), (i, y + h), col, 1)
                    if i == playhead and with_playhead:
                        line(layer, (i , 0), (i, self.height), self.white, 5)
        return layer

    def paste(self, layer, to_paste, coords = (0,0)):
        """
        Paste given set of pixels onto a layer

        Args:
            layer (np.array): The layer to draw to
            to_paste (np.array): The thing to paste
            coords (tuple): int coordinates of where to paste
        Returns:
            layer (np.array): The layer that has been updated 
        """
        x = coords[0]
        y = coords[1]
        w = to_paste.shape[1]
        h = to_paste.shape[0]
        cw = layer.shape[1]
        ch = layer.shape[0]
        if x + w <= cw and y + h <= ch and x >= 0 and y >= 0:
            layer[y:y+h,x:x+w] = to_paste
        return layer

    def update_canvas(self):
        """
        Render the layer stack onto the canvas
        """
        self.layers.insert(0, [self.canvas,1])
        for i in range(len(self.layers)-1):
            c1 = self.layers[i]
            c2 = self.layers[i+1]
            upper_layer_objects_mask = cv2.bitwise_not(cv2.inRange(cv2.cvtColor(c2[0], cv2.COLOR_BGR2GRAY), 1, 1))
            #Dont blend into parts of lower layer where there isnt stuff in the upper layer
            lower_hidden_by_upper = cv2.bitwise_and(c1[0], c1[0], mask=upper_layer_objects_mask)
            #Blend appropriate bits
            c2[0] = (c2[0]*c2[1]) + (lower_hidden_by_upper*(1-c2[1]))
            upper_layer_no_objects_mask = cv2.bitwise_not(upper_layer_objects_mask)
            lower_not_hidden_by_upper = cv2.bitwise_and(c1[0], c1[0], mask=upper_layer_no_objects_mask)
            #Add in blended stuff (not over unblended stuff)
            #Swap the 1s for 0s so we dont overflow
            c2[0][c2[0]==1] = 0
            c2[0] = np.array(c2[0] + lower_not_hidden_by_upper, dtype = np.uint8)
        self.canvas = self.layers[-1][0]
        self.layers = []

    def get_images(self, root_dir = "data/animal_thumbnails/land_mammals/cat", thumbnail_size = (50,50)):
        #Set the thumbnail size (you can change this but you won't want to make it too big!)
        images = []
        #Search through the separate file extensions 
        for ext in ('*.jpeg', '*.jpg', '*.png'):
            #Search through all the image files recursively in the directory
            for file in glob.glob(f'{root_dir}/**/{ext}', recursive=True):
                #Open the image using the file path
                im = cv2.imread(file)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                #Create a downsampled image based on the thumbnail size
                thumbnail = cv2.resize(im, thumbnail_size)
                thumbnail = np.asarray(thumbnail)
                #Check not grayscale (only has 2 dimensions)
                if len(thumbnail.shape) == 3:
                    #Append thumbnail to the list of all the images
                    #Drop any channels beyond rbg (e.g. Alpha for .png files)
                    images.append(thumbnail[:,:,:3])

        print(f'There have been {len(images)} images found')
        #Convert list of images to a numpy array
        image_set_array = np.asarray(images)
        return image_set_array

    def exit(self):
        self.music.stop()
        cv2.destroyAllWindows() 
        cv2.waitKey(1)
        sys.exit(0)
    
    def start_record(self):
        """
        Start collecting frames
        """
        if not self.recording:
            print("starting record")
            self.recording_buffer = []
            self.recording = True
    
    def stop_record(self, output_video_path = "output.mp4", fps = 25):
        """
        Stop collecting frames and render capture
        Args:
            output_video_path (str): where to save file
            fps (int): The frame rate to render the video. Defaults to 25
        """
        if self.recording:
            print("stopping record, writing file")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (self.width, self.height))
            frame_interval = (1.0 / fps) * 1000
            next_frame_time = 0.0

            for i in range(1, len(self.recording_buffer)):
                current_frame = self.recording_buffer[i]["frame"]
                current_time = self.recording_buffer[i]["timestamp"]
                
                while next_frame_time <= current_time:
                    print(next_frame_time, current_time)
                    out.write(current_frame) 
                    next_frame_time += frame_interval

            out.release()
            self.recording = False
            self.recording_buffer = []

    #Main drawing loop
    def start_loop(self, 
                   setup = lambda *args: None, 
                   draw = lambda *args: None
                   ):
        """
        Begin the drawing loop
        Args:
            setup (function): A function to call once at the beginning
            draw (function): A function to call on a loop
        """
        done = False
        setup()
        # Signal handler function
        def signal_handler(sig, frame):
            print('You pressed Ctrl+C! Closing the window.')
            self.exit()

        try:
            # Link the signal handler to SIGINT
            signal.signal(signal.SIGTSTP, signal_handler)
        except:
            pass

        name = "hold q to quit or ctrl z in terminal"
        cv2.namedWindow(name)
        cv2.setMouseCallback(name,self.mouse_moved)
        try :
            while not done:

                draw()                
                self.update_canvas()
                #Draw to window
                canvas_rgb = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
                cv2.imshow(name, canvas_rgb)
                
                if self.recording:
                    self.recording_buffer.append({"frame":canvas_rgb,"timestamp":self.millis})
                
                key = cv2.waitKey(1)
                if key & 0xFF == ord('p'): # print when 'p' is pressed
                    print("PRINT")
                    cv2.imwrite("screencap" + str(time.thread_time()) + ".png", cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB))
                elif key & 0xFF == ord('q'): # quit when 'q' is pressed
                    done = True
                    self.exit()
                    break
                elif not key == -1: 
                    self.on_key_pressed(chr(key & 0xFF))
            
                self.millis = int(round(time.time() * 1000)) - self.start_time_millis
                self.frame += 1

        except Exception as e:
            done = True
            print(e)
            traceback.print_exc()
            self.exit()            
        
        self.exit()

#Parent class for audio providers 
class AudioDevice:
    def __init__(self, on_new_frame = lambda n=1:0,
                 analyse=True, fft_size=1024, buffer_size=2048, sr=44100, output_device=None):
        self.running = True
        self.sr = sr
        self.fft_size = fft_size
        self.audio_latency = 5
        self.audio_buffer_write_ptr = 0
        self.fft_vals = [np.zeros((fft_size//2)+1) for i in range(self.audio_latency)]
        self.buffer_size = buffer_size
        self.amplitude = np.zeros(self.audio_latency)
        self.analyse = analyse
        self.output_device = output_device
        print(os.name)
        self.on_new_frame = on_new_frame
        self.internal_callback = lambda:0

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

            self.fft_vals[self.audio_buffer_write_ptr] = np.mean(np.abs(fft_results),axis=0)

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
            print(sd.query_devices())
            print("output_device set to default", sd.default.device[1])

        if self.output_device is not None:
            self.channels = sd.query_devices(self.output_device)['max_output_channels']
            print("channels:", self.channels)
        
        print("play_audio", "channels", self.channels, self.sr, "output_device",self.output_device)
        with sd.OutputStream(channels=self.channels, samplerate=self.sr, blocksize=self.buffer_size, device=self.output_device) as stream:
            while self.running:
                if not self.pause_event.is_set():
                    audio_data = self.audio_callback()
                    #duplicate to fill channels (mostly generating mono)
                    if audio_data.ndim < self.channels:
                        audio_data = np.tile(audio_data[:, None], (1, self.channels))
                    else:
                        audio_data = audio_data[np.newaxis, :]
                    # print(audio_data.shape, audio_data.ndim, self.channels, stream.channels)
                    stream.write(audio_data)
                    self.do_analysis(audio_data[:,0])
                else:
                    time.sleep(0.1)  
        
    def play(self):
        self.running = True
        self.play_thread.start()

    def pause(self):
        self.pause_event.set()

    def resume(self):
        self.pause_event.clear()

    def stop(self):
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

    def capture_audio(self):
        print("capture_audio (AudioCapture)", self.running, self.input_device, self.channels)
        
        with sd.InputStream(callback=self.audio_callback, 
                            channels=1, 
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
        self.current_sample = 0
    
    def audio_callback(self):
        if self.pause_event.is_set():
            return np.zeros(self.buffer_size) # Fill buffer with silence if paused
        else:
            audio_buffer = self.y[self.current_sample:self.current_sample +self.buffer_size]
            self.current_sample += self.buffer_size
            if self.current_sample > len(self.y):
                wrap_ptr = self.current_sample - len(self.y)
                wrap_signal = self.y[0:wrap_ptr]
                audio_buffer = np.concatenate((audio_buffer,wrap_signal))
                self.current_sample = wrap_ptr
            self.audio_buffer = audio_buffer
            self.internal_callback()
            self.on_new_frame(audio_buffer)
            return audio_buffer * self.gain

#Main class for music analysis and generation
class Audio:
    
    audio_outputs = []  

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
        return len(self.audio_outputs)-1

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
        return len(self.audio_outputs)-1
    
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
                        self.update_rave_latent(a.model.encode(input_audio))
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
        return len(self.audio_outputs)-1

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
        y, sr = librosa.load(file_path, sr=sr)
        return self.start_sample_stream(y, fft_size, buffer_size, sr, output_device, analyse)
    
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
        self.y = y
        self.sr = sr
        #Beat info
        self.tempo, self.beats = librosa.beat.beat_track(y=self.y, sr=self.sr, units='samples')
        self.beat_ptr = 0
        device = SamplePlayer(y = self.y, analyse=analyse,
                            fft_size = fft_size, buffer_size = buffer_size, sr = self.sr, output_device=output_device)
        self.audio_outputs.append(device)
        return len(self.audio_outputs)-1
    
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

    def play(self):
        for o in self.audio_outputs:
            o.play()

    def stop(self):
        for o in self.audio_outputs:
            o.stop()

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

