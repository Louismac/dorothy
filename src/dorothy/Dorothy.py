

import numpy as np
import moderngl_window as mglw
import glm
from typing import Tuple, Optional, Callable
import time
from .css_colours import css_colours
from .Audio import *
from .DorothyWindow import DorothyWindow
import cv2
import wave
import subprocess
import datetime
import sys
import traceback
from contextlib import contextmanager
from .DorothyShaders import DOTSHADERS
import signal


class Dorothy:

    # Class variables for window management
    _pending_instance = None
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance  
    
    def __init__(self, width: int = 800, height: int = 600, title: str = "Dorothy"):

        # Only init once
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        Dorothy._instance = self
        Dorothy._pending_instance = self
        self._persistent_canvas = None  # Will be created after renderer is ready
        self._auto_display_canvas = True
        
        # Window configuration
        self.window_size = (width, height)
        self.window_title = title
        
        # Renderer (will be initialized when window is created)
        self.renderer = None
        self.wnd = None
        self._initialized = False

        self.end_recording_at = np.inf
        self.recording = False
        self.should_clear = False
        
        # User sketch
        self.setup_fn = None
        self.draw_fn = None
        self.on_close = None
        self.on_mouse_press = None
        self.on_mouse_release = None
        self.on_mouse_drag = None
        self.on_scroll = None
        self.on_key_press = None
        
        # Processing-like properties
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_down = False
        self.frames = 0
        self.start_time = time.time()
        self.lfos = []

        self._colours = {name.replace(" ", "_"): rgb for name, rgb in css_colours.items()}
        print("done load colours")
        self._persistent_canvas = None

    def _ensure_persistent_canvas(self):
        """Create persistent canvas layer on first render"""
        if self._persistent_canvas is None and self.renderer:
            self._persistent_canvas = self.renderer.get_layer()
            # Start with transparent background
            self.renderer.begin_layer(self._persistent_canvas)
            self.renderer.ctx.clear(0.0, 0.0, 0.0, 0.0)
            self.renderer.end_layer()
    
    def background(self, color: Tuple):
        """Clear the active layer with a color"""
        layer_id = self.renderer.active_layer
        # print(f"clearing layer:{layer_id} with {color}" )
        if layer_id is None:
            # Not in a layer - shouldn't happen in normal use
            return
        if len(color) == 3:
            color = (*color, 255)
            
        if color[3] >= 255:  # Fully opaque - use fast clear
            color = self._parse_color(color)
            self.renderer.clear_layer(layer_id, color)
        else:  # Semi-transparent - draw rectangle
            camera = self.renderer.camera.mode
            self.camera_2d()
            self.fill(color)
            self.rectangle((0, 0), (self.width, self.height))
            self.renderer.camera.mode = camera

    def __getattr__(self, name):
        """Dynamically retrieve color attributes"""
        # Check if _colours exists first (in case called during __init__)
        if '_colours' in self.__dict__:
            if name in self._colours:
                return self._colours[name]
            # Provide helpful error for possible typos
            close_matches = [c for c in self._colours.keys() if c.startswith(name[:3])]
            if close_matches:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'. "
                    f"Did you mean one of: {', '.join(close_matches[:5])}?"
                )
        
        # Standard error message
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
    
        # Screen capture
    def get_pixels(self) -> np.ndarray:
        """Get current screen pixels as numpy array (for recording/screenshots)
        
        Returns:
            np.ndarray: BGR image array (height, width, 3) compatible with OpenCV
            
        Example:
            frame = dot.get_pixels()
            cv2.imwrite('screenshot.png', frame)
        """
        self._ensure_renderer()
        return self.renderer.get_pixels()
    
    def start_loop(self, setup_fn: Callable, draw_fn: Callable):
        """Start the render loop with setup and draw functions"""
        self.setup_fn = setup_fn
        self.draw_fn = draw_fn
        
        # Configure and run the window
        DorothyWindow.window_size = self.window_size
        DorothyWindow.title = self.window_title

        # Signal handler function
        def signal_handler(sig, frame):
            print('You pressed Ctrl+C! Closing the window.')
            self.exit()
            
        #only for unix like systems
        if hasattr(signal, 'SIGTSTP'):
            try:
                # Link the signal handler to SIGINT
                signal.signal(signal.SIGTSTP, signal_handler)
            except Exception as e:
                print(e)
                traceback.print_exc()
        else:
            print("SIGTSTP not available on this platform")

        
        
        # Run the window (this will call setup_fn when ready)
        try:
            mglw.run_window_config(DorothyWindow)
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.exit()  
    

    """
    Live coding implementation with debugging
    Add this to your Dorothy class
    """

    def start_livecode_loop(self, sketch_module):
        """Start a live coding loop that reloads code on file changes"""
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        import importlib
        import traceback
        import inspect
        from pathlib import Path
        import time
        
        sketch_file = Path(sketch_module.__file__)
        
        # Debug: Print the file we're watching
        print(f"🔍 DEBUG: Watching file: {sketch_file}")
        print(f"🔍 DEBUG: File exists: {sketch_file.exists()}")
        print(f"🔍 DEBUG: Watching dikrectory: {sketch_file.parent}")
        
        #Override the init in case someone is start_loop-ing by mistake
        def new_init(self):
            print("Overridden init")
        sketch_module.MySketch.__init__ = new_init
        my_sketch = sketch_module.MySketch()
        
        self.was_error = False
        self.reload_requested = False
        self.reload_count = 0
        
        class SketchReloadHandler(FileSystemEventHandler):
            def __init__(self, dorothy_instance):
                self.dorothy = dorothy_instance
                self.last_modified = 0
                self.event_count = 0
                
            def on_any_event(self, event):
                """Catch ALL events - modified, created, moved, etc."""
                self.event_count += 1
                
                # Debug: Print ALL file events
                print(f"🔍 DEBUG [{self.event_count}]: {event.event_type} event!")
                print(f"   Path: {event.src_path}")
                print(f"   Is directory: {event.is_directory}")
                
                if event.is_directory:
                    print(f"   ⏭️  Skipping (directory)")
                    return
                
                # Check if this is our sketch file (by name, not full path)
                file_path = Path(event.src_path)
                
                print(f"   File name: {file_path.name}")
                print(f"   Target name: {sketch_file.name}")
                print(f"   Match by name: {file_path.name == sketch_file.name}")
                
                # Match by filename (more robust for editor saves)
                if file_path.name == sketch_file.name and file_path.suffix == '.py':
                    current_time = time.time()
                    time_since_last = current_time - self.last_modified
                    
                    print(f"   ⏱️  Time since last reload: {time_since_last:.2f}s")
                    
                    # Debounce
                    if time_since_last < 0.5:
                        print(f"   ⏭️  DEBOUNCED (too soon, need 0.5s)")
                        return
                    
                    print(f"   ✅ SETTING reload_requested = True")
                    self.last_modified = current_time
                    self.dorothy.reload_requested = True
                else:
                    print(f"   ⏭️  Not our target file")
        
        def reload_sketch():
            """Reload the sketch without closing window"""
            try:
                print(f"📝 Reloading {sketch_file.name}...")
                self.reload_count += 1
                print(f"🔍 DEBUG: Reload count: {self.reload_count}")
                
                # Clear any cached modules
                if sketch_module.__name__ in sys.modules:
                    print(f"🔍 DEBUG: Removing {sketch_module.__name__} from sys.modules")
                
                # Reload the module
                importlib.reload(sketch_module)
                new_class = sketch_module.MySketch
                
                print(f"🔍 DEBUG: Old class: {my_sketch.__class__}")
                print(f"🔍 DEBUG: New class: {new_class}")
                
                # Update the class
                my_sketch.__class__ = new_class
                
                print(f"🔍 DEBUG: Updated class, calling setup()...")
                
                # Re-run setup
                my_sketch.setup()
                
                print(f"✅ Reloaded successfully!")
                self.was_error = False
                
            except Exception:
                if not self.was_error:
                    print("❌ Error reloading:")
                    print(traceback.format_exc())
                    self.was_error = True
        
        def setup_wrapper():
            """Initial setup"""
            print(f"🔍 DEBUG: setup_wrapper called")
            try:
                my_sketch.setup()
                self.was_error = False
            except Exception:
                if not self.was_error:
                    print("❌ Error in setup:")
                    print(traceback.format_exc())
                    self.was_error = True
        
        def draw_wrapper():
            """Draw loop with reload checking"""
            # Debug every 120 frames (every 2 seconds at 60fps)
            if self.frames % 120 == 0:
                print(f"🔍 DEBUG: Frame {self.frames}, reload_requested = {self.reload_requested}")
            
            # Check if reload was requested
            if self.reload_requested:
                print(f"\n🔥 DEBUG: RELOAD REQUESTED! Calling reload_sketch()")
                self.reload_requested = False
                reload_sketch()
            
            try:
                # Handle run_once function
                if hasattr(my_sketch, 'run_once'):
                    func_key = inspect.getsource(my_sketch.run_once)
                    if not hasattr(my_sketch, 'old_once_func'):
                        my_sketch.old_once_func = func_key
                    if my_sketch.old_once_func != func_key:
                        my_sketch.once_ran = False
                        my_sketch.old_once_func = func_key
                    if not getattr(my_sketch, 'once_ran', False):
                        my_sketch.run_once()
                        my_sketch.once_ran = True
                
                # Call draw
                my_sketch.draw()
                self.was_error = False
                
            except Exception:
                if not self.was_error:
                    print("❌ Error in draw:")
                    print(traceback.format_exc())
                    self.was_error = True
        
        # Start file watching
        print(f"🔍 DEBUG: Creating file observer...")
        event_handler = SketchReloadHandler(self)
        observer = Observer()
        observer.schedule(event_handler, path=str(sketch_file.parent), recursive=False)
        observer.start()
        print(f"🔍 DEBUG: Observer started successfully")
        
        print(f"\n👀 Watching {sketch_file.name} for changes...")
        print(f"💡 Edit and save {sketch_file.name} to see live updates!")
        print(f"🎨 Press Q or ESC to quit\n")  
        
        try:
            # This is the ONLY start_loop call
            self.start_loop(setup_wrapper, draw_wrapper)
        finally:
            print(f"🔍 DEBUG: Stopping observer...")
            observer.stop()
            observer.join()
            print(f"🔍 DEBUG: Observer stopped")
    
    def draw_playhead(self, audio_output=0):
        if audio_output < len(self.music.audio_outputs):
            output = self.music.audio_outputs[audio_output]
            if isinstance(output, SamplePlayer):
                latency = output.buffer_size * output.audio_latency
                mixed = output.y.mean(axis=0)
                playhead = int(((output.current_sample-latency) / len(mixed)) *self.width)
                self.fill(self.white)
                self.set_stroke_weight(5)
                self.line((playhead , 0), (playhead, self.height))

    def draw_waveform(self, audio_output = 0, col=None, with_playhead = False):
        """
        Draw current waveform loaded into given audio output to layer

        Args:
            audio_output (int): The index of the audio device to show
            col (tuple): RGB Colour to draw the waveform. Defaults to black.
            with_playhead (bool): Show current position in audio. Defaults to false.
        Returns:
            layer (np.array): The layer that has been updated 
        """
        if audio_output < len(self.music.audio_outputs):
            output = self.music.audio_outputs[audio_output]
            if isinstance(output, SamplePlayer):
                mixed = output.y.mean(axis=0)
                samples_per_pixel = len(mixed) / self.width
                self.stroke(col)
                self.set_stroke_weight(2)
                for i in range(self.width):
                    val = mixed[int(samples_per_pixel*i)]
                    h = int(val * self.height)
                    y = self.height//2 - h//2
                    if not col == None:
                        self.line((i, y), (i, y + h))
                if with_playhead:
                    self.draw_playhead(audio_output)
                    

    def get_images(self, root_dir = "data/animal_thumbnails/land_mammals/cat", thumbnail_size = (50,50)):
        #Set the thumbnail size (you can change this but you won't want to make it too big!)
        images = []
        import glob 
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
        self.music.clean_up()

    def start_record(self, audio_output=0, end=None):
        """
        Start collecting frames
        """
        if not self.recording:
            print("starting record", self.millis, end)
            self.video_recording_buffer = []
            self.start_record_time = self.millis
            if audio_output < len(self.music.audio_outputs):
                print("starting record audio", audio_output)
                output = self.music.audio_outputs[audio_output]
                output.recording_buffer = []
                output.recording = True
            self.recording = True
            if not end == None:
                self.end_recording_at = self.millis + end
            
    def stop_record(self, output_video_path = None, fps = 25, audio_output = 0, audio_latency = 6):
        """
        Stop collecting frames and render capture
        Args:
            output_video_path (str): where to save file
            fps (int): The frame rate to render the video. Defaults to 25
            audio_output (int): which audio device to use
            audio_latency (int): number of frames to pad to resync audio with video
        """
        if output_video_path is None:
            output_video_path = "record_output_" + datetime.datetime.now().strftime("%Y_%m_%d_%h_%H_%M_%S") + ".mp4"
        if self.recording:
            print("stopping record, writing file")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (self.width, self.height))
            frame_interval = (1.0 / fps) * 1000
            next_frame_time = self.start_record_time

            for i in range(1, len(self.video_recording_buffer)):
                current_frame = self.video_recording_buffer[i]["frame"]
                current_time = self.video_recording_buffer[i]["timestamp"]
                
                while next_frame_time <= current_time:
                    out.write(current_frame) 
                    next_frame_time += frame_interval

            out.release()
            self.recording = False
            self.video_recording_buffer = []

            if audio_output < len(self.music.audio_outputs):
                output = self.music.audio_outputs[audio_output]
                
                def save_audio_to_wav(audio_frames, sample_rate, file_name):
                    audio_frames = np.array(audio_frames)
                    print("Before reshape:", np.array(audio_frames).shape)
                    audio_frames_flat = audio_frames.reshape(-1, audio_frames.shape[-1])
                    print("After reshape:", audio_frames_flat.shape)
                    with wave.open(file_name, 'wb') as wav_file:
                        wav_file.setnchannels(audio_frames_flat.shape[1]) # should be stereo
                        wav_file.setsampwidth(2)  # 16-bit audio
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes((audio_frames_flat * 32767).astype(np.int16).tobytes())

                def combine_audio_video(wav_file, mp4_file, output_file):
                    command = [
                        'ffmpeg', '-y', '-i', mp4_file, '-i', wav_file,
                        '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', output_file
                    ]
                    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if result.returncode == 0:
                        print(f"Successfully combined audio and video into {output_file}")
                        os.remove(wav_file)
                        os.remove(mp4_file)
                        os.rename(output_file, mp4_file)
                        print(f"Renamed {output_file} to {mp4_file}")
                    else:
                        print(f"Error combining audio and video: {result.stderr}")
                
                #mono
                audio_data = np.array(output.recording_buffer)
                print(audio_data.shape)
                if len(audio_data) > 0:
                    combined_file = 'combined_video.mp4'
                    sample_rate = output.sr  
                    audio_file = 'output_audio.wav'
                    
                    #padd some zeros to get back in sync with visuals (audio is early apparently)
                    audio_data = np.pad(audio_data, ((audio_latency,0),(0,0), (0, 0)), mode='constant', constant_values=0)
                    save_audio_to_wav(audio_data, sample_rate, audio_file)
                    combine_audio_video(audio_file, output_video_path, combined_file)
                    output.audio_recording_buffer = []
                    output.recording = False
                else:
                    print("no audio to write to file")

    def get_lfo(self, osc='sine', freq=1.0, range=(0, 1)):
        """Create a Low Frequency Oscillator for parameter modulation
        
        Args:
            osc: Oscillator type - 'sine', 'saw', 'square', 'triangle'
            freq: Frequency in Hz (cycles per second)
            range: Tuple of (min, max) to map oscillator output to
        
        Returns:
            int: LFO ID to use with lfo_value()
        
        Example:
            # Create LFOs in setup
            self.size_lfo = dot.get_lfo('sine', freq=0.5, range=(20, 100))
            self.color_lfo = dot.get_lfo('saw', freq=2.0, range=(0, 255))
            
            # Use in draw
            size = dot.lfo_value(self.size_lfo)
            dot.circle((200, 200), size)
        """
        self._ensure_renderer()
        
        if not hasattr(self, 'lfos'):
            self.lfos = []
        
        self.lfos.append({
            "phase": 0.0,
            "freq": freq,
            "osc": osc,
            "range": range,
            "start": self.millis
        })
        
        return len(self.lfos) - 1

    def update_lfos(self):
        """Update all LFO phases - called automatically from on_render"""
        if not hasattr(self, 'lfos'):
            return
        
        for lfo in self.lfos:
            # Calculate elapsed time in seconds
            elapsed = (self.millis - lfo['start']) / 1000.0
            
            # Update phase based on frequency
            # phase = (frequency * time) mod 1.0 (keeps phase in 0-1 range)
            lfo['phase'] = (lfo['freq'] * elapsed) % 1.0

    def lfo_value(self, lfo_id=0):
        """Get current value of an LFO
        
        Args:
            lfo_id: ID returned from get_lfo()
        
        Returns:
            float: Current LFO value mapped to specified range
        
        Example:
            size = dot.lfo_value(self.size_lfo)
            dot.circle((200, 200), size)
        """
        if not hasattr(self, 'lfos') or lfo_id >= len(self.lfos):
            return 0.0
        
        lfo = self.lfos[lfo_id]
        phase = lfo['phase']
        osc_type = lfo['osc']
        min_val, max_val = lfo['range']
        
        # Generate oscillator value based on type (-1 to 1)
        if osc_type == 'sine':
            # Sine wave: smooth oscillation
            raw = np.sin(phase * 2 * np.pi)
            
        elif osc_type == 'saw':
            # Sawtooth wave: linear ramp from -1 to 1
            raw = (phase * 2) - 1
            
        elif osc_type == 'square':
            # Square wave: alternates between -1 and 1
            raw = 1.0 if phase < 0.5 else -1.0
            
        elif osc_type == 'triangle':
            # Triangle wave: linear ramp up then down
            if phase < 0.5:
                raw = (phase * 4) - 1  # 0->0.5 maps to -1->1
            else:
                raw = 3 - (phase * 4)  # 0.5->1 maps to 1->-1
        else:
            raw = 0.0
        
        # Map from -1..1 to min_val..max_val
        normalized = (raw + 1.0) / 2.0  # -1..1 to 0..1
        mapped = min_val + (normalized * (max_val - min_val))
        
        return mapped
    # ====== Properties ======
    
    @property
    def width(self) -> int:
        return self.window_size[0]
    
    @property
    def height(self) -> int:
        return self.window_size[1]
    
    @property
    def centre(self) -> Tuple:
        return (self.width//2, self.height/2)
    
    @property
    def millis(self) -> float:
        """Time in milliseconds since start"""
        return (time.time() - self.start_time) * 1000
    
    # ====== Drawing API (delegates to renderer) ======
    
    def _ensure_renderer(self):
        """Ensure renderer is initialized"""
        if not self._initialized:
            raise RuntimeError("Dorothy not initialized. Call start_loop() first.")
    
    def _parse_color(self, color):
        """Parse color from various formats to normalized (0-1) values"""
        if isinstance(color, tuple):
            if len(color) == 3:
                r, g, b = color
                return (r/255.0, g/255.0, b/255.0, 1.0)
            elif len(color) == 4:
                r, g, b, a = color
                return (r/255.0, g/255.0, b/255.0, a/255.0)
        return color

    def fill(self, color: Tuple):
        """Set fill color"""
        self._ensure_renderer()
        self.renderer.fill(color)
    
    def no_fill(self):
        """Disable fill"""
        self._ensure_renderer()
        self.renderer.no_fill()
    
    def stroke(self, color: Tuple):
        """Set stroke color"""
        self._ensure_renderer()
        self.renderer.stroke(color)
    
    def no_stroke(self):
        """Disable stroke"""
        self._ensure_renderer()
        self.renderer.no_stroke()
    
    def set_stroke_weight(self, weight: float):
        """Set stroke weight"""
        self._ensure_renderer()
        self.renderer.set_stroke_weight(weight)
    
    # 2D shapes
    def circle(self, center: Tuple[float, float], radius: float, annotate: bool = False):
        """Draw a circle"""
        self._ensure_renderer()
        self.renderer.circle(center, radius, annotate)
    
    def rectangle(self, pos1: Tuple[float, float], pos2: Tuple[float, float], annotate: bool = False):
        """Draw a rectangle"""
        self._ensure_renderer()
        self.renderer.rectangle(pos1, pos2, annotate)
    
    def line(self, pos1: Tuple[float, float], pos2: Tuple[float, float], annotate: bool = False):
        """Draw a line"""
        self._ensure_renderer()
        self.renderer.line(pos1, pos2, annotate)
    
    def polyline(self, points, closed: bool = False):
        """Draw a circle"""
        self._ensure_renderer()
        self.renderer.polyline(points, closed)

    def polygon(self, points):
        """Draw a circle"""
        self._ensure_renderer()
        self.renderer.polygon(points)

   # 3D shapes
    def sphere(self, radius: float = 1.0, position: Tuple[float, float, float] = (0, 0, 0)):
        """Draw a 3D sphere
        
        Args:
            radius: Sphere radius
            position: (x, y, z) center position
        """
        self._ensure_renderer()
        self.renderer.sphere(radius, position)
    
    def box(self, size: Tuple[float, float, float] = (1.0, 1.0, 1.0), 
            position: Tuple[float, float, float] = (0, 0, 0)):
        """Draw a 3D box
        
        Args:
            size: (width, height, depth) tuple
            position: (x, y, z) center position
        """
        self._ensure_renderer()
        self.renderer.box(size, position)
    
    # Transforms
    @contextmanager
    def transform(self):
        """Context manager for transforms"""
        self._ensure_renderer()
        self.renderer.push_matrix()
        try:
            yield
        finally:
            self.renderer.pop_matrix()
    
    def translate(self, x: float, y: float, z: float = 0):
        """Translate"""
        self._ensure_renderer()
        self.renderer.translate(x, y, z)
    
    def rotate(self, angle: float, x: float = 0, y: float = 0, z: float = 1):
        """Rotate (angle in radians)"""
        self._ensure_renderer()
        self.renderer.rotate(angle, x, y, z)
    
    def scale(self, x: float, y:float=None):
        """Scale 2D"""
        self._ensure_renderer()
        if y is None:
            y = x
        self.renderer.scale(x,y)
    
    def reset_transforms(self):
        """Reset transforms"""
        self._ensure_renderer()
        self.renderer.reset_transforms()
    
    # Camera
    def camera_3d(self):
        """Switch to 3D camera mode"""
        self._ensure_renderer()
        self.renderer.camera.mode = '3d'
    
    def camera_2d(self):
        """Switch to 2D camera mode (orthographic)"""
        self._ensure_renderer()
        self.renderer.camera.mode = '2d'
    
    def set_camera(self, eye: Tuple[float, float, float], 
                   target: Tuple[float, float, float] = (0, 0, 0)):
        """Set camera position and target"""
        self._ensure_renderer()
        self.renderer.camera.position = glm.vec3(*eye)
        self.renderer.camera.target = glm.vec3(*target)
    
    # Layers
    def get_layer(self) -> int:
        """Create a new rendering layer (offscreen framebuffer)
        
        Returns:
            layer_id: Unique identifier for this layer
            
        """
        self._ensure_renderer()
        return self.renderer.get_layer()
    
    @contextmanager
    def layer(self, layer_id):
        """Context manager for drawing to a layer"""
        self._ensure_renderer()
        self.renderer.begin_layer(layer_id)
        try:
            yield
        finally:
            self.renderer.end_layer()
    
    def draw_layer(self, layer_id: int, alpha: float = 1.0):
        """Draw a layer to the screen with transparency
        
        Args:
            layer_id: The layer to draw
            alpha: Transparency (0.0 = invisible, 1.0 = opaque)
        """
        self._ensure_renderer()
        self.renderer.draw_layer(layer_id, alpha)
    
    def release_layer(self, layer_id: int):
        """Free a layer's resources when no longer needed
        
        Args:
            layer_id: The layer to release
        """
        self._ensure_renderer()
        self.renderer.release_layer(layer_id)

    def pixelate(self, pixel_size=8.0, accumulate=False):
        """Apply pixelation effect
        
        Args:
            pixel_size: Size of pixels (larger = more pixelated)
            accumulate: If True, effect accumulates; if False, just display filter
        """
        self._ensure_renderer()
        self.apply_shader(DOTSHADERS.PIXELATE, 
                            accumulate=accumulate,
                            pixelSize=pixel_size
                        )

    def blur(self, accumulate=False):
        """Apply blur effect"""
        self._ensure_renderer()
        self.apply_shader(DOTSHADERS.BLUR, accumulate)

    def rgb_split(self, offset=0.01, accumulate=False):
        """Apply RGB split/glitch effect
        
        Args:
            offset: How far to split RGB channels (0.0-0.1)
            accumulate: If True, effect accumulates
        """
        self._ensure_renderer()
        self.apply_shader(DOTSHADERS.RGB_SPLIT,
                          accumulate=accumulate, 
                            offset=offset 
                            )
        
    def feedback(self, zoom=0.98, accumulate=True):
        """Apply RGB split/glitch effect
        
        Args:
            offset: How far to split RGB channels (0.0-0.1)
            accumulate: If True, effect accumulates
        """
        self._ensure_renderer()
        self.apply_shader(DOTSHADERS.FEEDBACK, 
                          accumulate=accumulate,
                          zoom=zoom
                        )

    def roll(self, offset_x=0.0, offset_y=0.0, accumulate=True):
        """Roll/shift the canvas with wrapping
        
        Args:
            offset_x: Horizontal shift in pixels
            offset_y: Vertical shift in pixels
            accumulate: Usually True for rolling effects
        """
        self._ensure_renderer()
        self.apply_shader(DOTSHADERS.ROLL,
                           accumulate = accumulate,
                           offset=(offset_x, offset_y))
        
    def invert(self, accumulate=False):
        """Invert colors"""
        self._ensure_renderer()
        self.apply_shader(DOTSHADERS.INVERT,accumulate=accumulate)
    
    def tile(self, grid_x=2, grid_y=2, accumulate=False):
        """Tile/repeat the canvas in a grid
        
        Args:
            grid_x: Number of tiles horizontally (default: 2)
            grid_y: Number of tiles vertically (default: 2)
            accumulate: If True, effect accumulates; if False, just display filter
        
        """
        self._ensure_renderer()
        self.apply_shader(
            DOTSHADERS.TILE,
            accumulate = accumulate,
            grid_size = (float(grid_x), float(grid_y))
        )

    def cutout(self, color, threshold=0.1, accumulate=True):
        """Make pixels of a specific color transparent
        
        Args:
            color: RGB tuple (0-255) or color constant to cut out
            threshold: How close colors need to match (0.0 = exact, 0.5 = loose)
            accumulate: If True, effect accumulates (default: True)

        """
        self._ensure_renderer()
        
        # Parse color and normalize to 0-1
        parsed = self._parse_color(color)
        normalized = (parsed[0], parsed[1], parsed[2])  # RGB only, no alpha
        
        self.apply_shader(
            DOTSHADERS.CUTOUT,
            accumulate=accumulate,
            cutout_color=normalized,
            threshold=float(threshold)
        )

    def apply_shader(self, fragment_shader_code: str, accumulate: bool = True, **uniforms):
        """Apply a custom post-processing shader to the canvas
        
        Args:
            fragment_shader_code: GLSL fragment shader source code
            accumulate: If True, shader effects build up over frames (feedback)
                    If False, shader is just a display filter (post-processing)
            **uniforms: Additional uniforms to pass to the shader
        
        """
        self._ensure_renderer()
        result = self.renderer.apply_shader(fragment_shader_code, uniforms, accumulate)
        
        # Store for on_render to display
        if result is not None:
            self._non_accumulating_shader_output = result
        else:
            self._non_accumulating_shader_output = None
    
    # Images
    def paste(self, image: np.ndarray, position: Tuple[int, int], 
              size: Optional[Tuple[int, int]] = None, alpha: float = 1.0):
        """Paste a numpy array (image) onto the canvas
        
        Args:
            image: NumPy array of pixels. Supports:
                   - (H, W, 3) for RGB
                   - (H, W, 4) for RGBA  
                   - (H, W) for grayscale
                   Values can be uint8 (0-255) or float (0.0-1.0)
            position: (x, y) top-left corner position
            size: Optional (width, height) to resize. None = original size
            alpha: Overall transparency (0.0-1.0)
            
        """
        self._ensure_renderer()
        self.renderer.paste(image, position, size, alpha)