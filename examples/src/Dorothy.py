import numpy as np
import cv2
from cv2 import rectangle
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

#For Rave example, not normally needed
try:
    import torch
    torch.set_grad_enabled(False)
    from utils.magnet import preprocess_data, RNNModel, generate
except ImportError:
    print("torch not available, machine learning examples won't work, otherwise ignore.")

#Parent class for audio providers 
class AudioDevice:
    def __init__(self, on_analysis_complete = lambda:0, on_new_frame = lambda n=1:0,
                 analyse=True, fft_size=1024, buffer_size=2048, sr=44100, output_device=None):
        self.running = True
        self.sr = sr
        self.fft_size = fft_size
        self.fft_vals = np.zeros((fft_size//2)+1)
        self.buffer_size = buffer_size
        self.amplitude = 0
        self.analyse = analyse
        self.output_device = output_device
        print(os.name)
        self.on_analysis_complete = on_analysis_complete
        self.on_new_frame = on_new_frame
        self.internal_callback = lambda:0
        if os.name == "posix":
            p = psutil.Process(os.getpid())
            p.nice(10)
        elif os.name == "nt":
            thread_id = threading.get_native_id()
            ctypes.windll.kernel32.SetThreadPriority(thread_id, 2)
        sd.default.samplerate = self.sr
        self.channels = 1
        print("output_device", output_device)
        if output_device is not None:
            self.channels = sd.query_devices(output_device)['max_output_channels']
        self.pause_event = threading.Event()
        self.play_thread = threading.Thread(target=self.capture_audio)
        self.gain = 1

    def do_analysis(self, audio_buffer):
        if self.analyse:
            #Get amplitude
            self.amplitude = np.mean(audio_buffer**2)
            num_frames = 1 + (len(audio_buffer) - self.fft_size) // self.fft_size//2
            fft_results = np.zeros((num_frames, self.fft_size), dtype=complex)
            window = np.hanning(self.fft_size)
            for i in range(num_frames):
                frame_start = i * self.fft_size//2
                frame_end = frame_start + self.fft_size
                frame = audio_buffer[frame_start:frame_end]
                windowed_frame = frame * window
                fft_results[i] = np.fft.fft(windowed_frame)

            self.fft_vals = np.abs(fft_results)
            # return the mean for visualising 
            self.on_analysis_complete(np.mean(self.fft_vals, axis=0), self.amplitude)

    #stub (overwritten in subclass)
    def audio_callback(self):
        self.on_new_frame()
        self.internal_callback()
        return np.zeros(self.buffer_size) # Fill buffer with silence
        
    def capture_audio(self):
        print("play_audio", "channels", self.channels, self.sr, "output_device",self.output_device)
        with sd.OutputStream(channels=self.channels, samplerate=self.sr, blocksize=self.buffer_size, device = self.output_device) as stream:
            while self.running:
                if not self.pause_event.is_set():
                    audio_data = self.audio_callback()
                    #duplicate to fill channels (mostly generating mono)
                    if audio_data.ndim < self.channels:
                        audio_data = np.tile(audio_data[:, None], (1, self.channels))
                    if self.output_device is not None:
                        # print(audio_data.shape, audio_data.ndim, self.channels, stream.channels)
                        stream.write(audio_data)
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

#Generating audio from RAVE models https://github.com/acids-ircam/RAVE
class MAGNetPlayer(AudioDevice):
    def __init__(self, model_path, dataset_path, **kwargs):
        super().__init__(**kwargs)
        
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision('high')
        self.device = torch.device('cpu')
        
        self.x_frames = self.load_dataset(dataset_path)
        self.model = self.load_model(model_path)
        
        self.ptr = 0
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
            audio_buffer = self.current_buffer[self.ptr:self.ptr +self.buffer_size]
            self.ptr += self.buffer_size
            #Currently dont do proper wrapping for buffer sizes that arent factors of self.frame_size
            if self.ptr >= self.frame_size:
                self.ptr = 0
                self.current_buffer = self.next_buffer.copy()
                if not self.generate_thread.is_alive():
                    self.generate_thread = threading.Thread(target=self.fill_next_buffer)
                    self.generate_thread.start()
            self.do_analysis(audio_buffer)
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
        self.ptr = 0
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
            audio_buffer = self.current_buffer[self.ptr:self.ptr +self.buffer_size]
            self.ptr += self.buffer_size
            #Currently dont do proper wrapping for buffer sizes that arent factors of self.frame_size
            if self.ptr >= self.frame_size:
                self.ptr = 0
                self.current_buffer = self.next_buffer.copy()
                if not self.generate_thread.is_alive():
                    self.generate_thread = threading.Thread(target=self.fill_next_buffer)
                    self.generate_thread.start()
            self.do_analysis(audio_buffer)
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
            self.do_analysis(self.audio_buffer)
            self.internal_callback()
            self.on_new_frame(self.audio_buffer)

    def capture_audio(self):
        print("capture_audio (AudioCapture)", self.running, self.input_device)
        
        with sd.InputStream(callback=self.audio_callback, 
                            channels=1, 
                            blocksize=self.buffer_size, 
                            samplerate=self.sr,
                            device = self.input_device):
            while self.running:
                # Just sleep and let the callback do all the work
                time.sleep(0.1)

#Class for playing back audio files
class FilePlayer(AudioDevice):

    def __init__(self, y=[0], **kwargs):
        super().__init__(**kwargs)
        self.y = y
        self.ptr = 0
    
    def audio_callback(self):
        if self.pause_event.is_set():
            return np.zeros(self.buffer_size) # Fill buffer with silence if paused
        else:
            audio_buffer = self.y[self.ptr:self.ptr +self.buffer_size]
            self.ptr += self.buffer_size
            if self.ptr > len(self.y):
                wrap_ptr = self.ptr - len(self.y)
                wrap_signal = self.y[0:wrap_ptr]
                audio_buffer = np.concatenate((audio_buffer,wrap_signal))
                self.ptr = wrap_ptr
            self.do_analysis(audio_buffer)
            self.audio_buffer = audio_buffer
            self.internal_callback()
            self.on_new_frame(audio_buffer)
            return audio_buffer * self.gain

#Main class for music analysis
class MusicAnalyser:
    
    audio_outputs = []
    fft_vals = np.zeros(2048)
    amplitude = 0

    def __init__(self):
        print("init")
 
    def on_analysis_complete(self, fft_vals, amplitude):
        # print(fft_vals, amplitude)
        self.fft_vals = fft_vals
        self.amplitude = amplitude

    def start_magnet_stream(self, model_path, dataset_path, buffer_size=2048, sr = 44100, output_device=None):
        device = MAGNetPlayer(model_path, dataset_path,
                            on_analysis_complete = self.on_analysis_complete, 
                            buffer_size=buffer_size, 
                            sr=sr,output_device = output_device)
        self.audio_outputs.append(device)
        return len(self.audio_outputs)-1

    def start_rave_stream(self, model_path="vintage.ts",fft_size=1024, buffer_size=2048, sr = 44100, latent_dim=16, output_device=None):
        device = RAVEPlayer(model_path=model_path, 
                                            on_analysis_complete = self.on_analysis_complete, 
                                            buffer_size=buffer_size, 
                                            sr=sr, fft_size=fft_size, 
                                            latent_dim=latent_dim,
                                            output_device = output_device)
        self.audio_outputs.append(device)
        return len(self.audio_outputs)-1
    
    def update_rave_from_stream(self, input=0):
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

    def start_device_stream(self, device, fft_size=1024, buffer_size=2048, sr = 44100, output_device=None, analyse=True):
        print(sd.query_devices(device))
        sd.default.device = device
        self.audio_outputs.append(AudioCapture(on_analysis_complete = self.on_analysis_complete, analyse=analyse,
                                          buffer_size=buffer_size, sr=sr, fft_size=fft_size, input_device=device))
        return len(self.audio_outputs)-1

    def start_file_stream(self, file_path, fft_size=1024, buffer_size=2048, sr = 44100, output_device=None, analyse = True):
        #load file
        self.y, self.sr = librosa.load(file_path, sr=sr)
        self.ptr = 0
        #Beat info
        self.tempo, self.beats = librosa.beat.beat_track(y=self.y, sr=self.sr, units='samples')
        self.beat_ptr = 0
        
        device = FilePlayer(y = self.y, on_analysis_complete = self.on_analysis_complete, analyse=analyse,
                                        fft_size = fft_size, buffer_size = buffer_size, sr = self.sr, output_device=output_device)
        self.audio_outputs.append(device)
        return len(self.audio_outputs)-1

    def play(self):
        for o in self.audio_outputs:
            o.play()

    def stop(self):
        for o in self.audio_outputs:
            o.stop()

    #Has there been a beat since this was last called?
    def is_beat(self):
        next_beat = self.beats[self.beat_ptr%len(self.beats)]
        is_beat = False
        if next_beat < self.ptr:
            is_beat = True
            self.beat_ptr += 1
        return is_beat
    
    def update_rave_latent(self, new_latent):
        for a in self.audio_outputs:
            if isinstance(a,RAVEPlayer):
                if isinstance(new_latent, torch.Tensor):
                    if a.current_latent.shape == new_latent.shape:
                        a.current_latent = new_latent
                    else:
                        print(f'new latent shape {new_latent.shape} does not match shape of existing latent {self.audio_outputs[0].current_latent.shape} ')
                else:
                    print(f'the latent passed into this function should be a Torch tensor, not a {type(new_latent)}')
      
#Main drawing class
class Dorothy:
    
    width = 640
    height = 480
    frame = 0
    mouse_x = 1
    mouse_y = 1
    mouse_down = False
    start_time_millis = int(round(time.time() * 1000))
    millis = 0
    layers = []
    music = MusicAnalyser()
    recording = False

    def __init__(self, width = 640, height = 480):
        self.width = width
        self.height = height
        self.canvas = np.ones((height,width,3), np.uint8)*255

    #Get a new layer for drawing
    def push_layer(self):
        return np.zeros((self.height,self.width,3), np.uint8)
    
    #Push layer back onto stack
    def pop_layer(self, c):
        self.layers.append([c,1])
        self.update_canvas()

    #Get a new layer for transparency drawing
    def to_alpha(self, alpha=1):
        new_canvas = np.zeros((self.height,self.width,3), np.uint8)
        self.layers.append([new_canvas,alpha])
        return self.layers[-1][0]
    
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
        return self.linear_transformation(canvas, m, origin)
    
    #Scale canvas given x and y factors and an origin
    def scale(self, canvas, sx=1, sy=1, origin =(0,0)):
        m = np.array([[sx,0.0],
                          [0.0,sy]])
        return self.transform(canvas, m, origin)
    
    #Rotate given theta and an origin
    def rotate(self, canvas, theta, origin = (0,0)):
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

    #Draw background
    def background(self, col, alpha=None):
        canvas = self.canvas
        #if black make slightly lighter
        if col == (0,0,0):
            col = (1,1,1)
        if not alpha is None:
            canvas = self.to_alpha(alpha)
        rectangle(canvas, (0,0), (self.width,self.height), col, -1)

    #Paste image
    def paste(self, canvas, to_paste, coords = (0,0)):
        x = coords[0]
        y = coords[1]
        w = to_paste.shape[1]
        h = to_paste.shape[0]
        cw = canvas.shape[1]
        ch = canvas.shape[0]
        if x + w <= cw and y + h <= ch and x >= 0 and y >= 0:
            canvas[y:y+h,x:x+w] = to_paste
        return canvas

    #Some complicated stuff to try and do alpha blending
    def update_canvas(self):
        self.layers.insert(0, [self.canvas,1])
        for i in range(len(self.layers)-1):
            c1 = self.layers[i]
            c2 = self.layers[i+1]
            
            _,mask = cv2.threshold(cv2.cvtColor(c2[0], cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
            #Dont blend into parts of lower layer where there isnt stuff in the upper layer
            masked_image = cv2.bitwise_and(c1[0], c1[0], mask=mask)
            #Blend appropriate bits
            c2[0] = (c2[0]*c2[1]) + (masked_image*(1-c2[1]))
            inverted_mask = cv2.bitwise_not(mask)
            inverted_masked = cv2.bitwise_and(c1[0], c1[0], mask=inverted_mask)
            #Add in blended stuff (not over unblended stuff)
            c2[0] = np.array(c2[0] + inverted_masked, dtype = np.uint8)
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
    
    def start_record(self, fps=40):
        output_video_path = "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_video_path, fourcc, fps, (self.width, self.height))
        self.recording = True
    
    def stop_record(self):
        print("stopping record, writing file")
        self.out.release()
        self.recording = False

    #Main drawing loop
    def start_loop(self, 
                   setup = lambda *args: None, 
                   draw = lambda *args: None
                   ):
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
                self.millis = int(round(time.time() * 1000)) - self.start_time_millis
                canvas_rgb = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
                #Draw to window
                cv2.imshow(name, canvas_rgb)
                if self.recording:
                    self.out.write(canvas_rgb)

                if cv2.waitKey(1) & 0xFF==ord('p'): # print when 'p' is pressed
                    print("PRINT")
                    cv2.imwrite("screencap" + str(time.thread_time()) + ".png", cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB))

                elif cv2.waitKey(1) & 0xFF==ord('q'): # quit when 'q' is pressed
                    done = True
                    self.exit()
                    break
            
                self.frame += 1
        except Exception as e:
            done = True
            print(e)
            traceback.print_exc()
            self.exit()            
        
        self.exit()