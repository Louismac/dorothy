import numpy as np
import cv2
import signal
import sys
import numpy as np
import time
import traceback
import glob
from .css_colours import css_colours
from .Audio import *
from time import sleep
import wave
import subprocess

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
    start_time_millis = 0
    millis = 0
    layers = []
    on_key_pressed = lambda *args: None
    recording = False
    video_recording_buffer = []
    start_record_time = 0
    test = 0
    fps = 10000
    current_transform = None

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
        self.music = Audio()
        self.end_recording_at = np.inf
        self.stroke_colour = None
        self.fill_colour = None
        self.stroke_weight = 1
        self.text_colour = (255,255,255)
        print("load colours")
        self._colours = {name.replace(" ", "_"): rgb for name, rgb in css_colours.items()}
        print("done load colours")

    def __getattr__(self, name):
        # Dynamically retrieve colour attributes
        try:
            return self._colours[name]
        except KeyError:
            raise AttributeError(f"{name} not found in colour attributes")

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
    
    #wrappers for opencv drawing function
    def stroke(self, stroke = (0,0,0)):
        """
        Set stroke colour
        
        Args:
            stroke (np.array): RGB colour
        """
        stroke = (int(stroke[0]),int(stroke[1]),int(stroke[2]))
        self.stroke_colour = stroke
        if self.fill_colour == None:
            self.text_colour = stroke

    def no_stroke(self):
        """
        Turn off stroke (borders)
        """
        self.stroke_colour = None
        self.text_colour = self.white
    
    def fill(self, fill = (0,0,0)):
        """
        Set fill colour
        
        Args:
            fill (np.array): RGB colour
        """
        fill = (int(fill[0]),int(fill[1]),int(fill[2]))
        self.fill_colour = fill
        self.text_colour = self.white

    def no_fill(self):
        """
        Turn off fill 
        """
        self.fill_colour = None
        self.text_colour = self.white
    
    def set_stroke_weight(self, stroke_weight=1):
        """
        Set stroke weight
        
        Args:
            stroke_weight (int): stroke weight
        """
        self.stroke_weight = int(stroke_weight)

    def circle(self, centre = (0,0), radius = 100, layer = None, annotate = False):
        """
        Draw circle
        
        Args:
            centre (np.array): Coordinates of centre
            radius (int): radius of circle
            layer (np.array): Where to draw, defaults to dot.canvas
            annotate (bool): If true, dimensions are annotated on sketch (for debug)
        """
        if layer is None:
            layer = self.canvas
            
        is_ellipse = False
        if not self.current_transform is None:
            centre = self.shift_coords(centre)
            eigenvalues, _ = np.linalg.eig(self.current_transform[:2,:2])
            if np.isclose(eigenvalues[0], eigenvalues[1]):
                radius *= eigenvalues[0]
                radius = int(radius)
            else:
                radius_vector_x = np.array([radius, centre[0]])
                radius_vector_y = np.array([centre[1], radius])
                radius_vector_x = self.shift_coords(radius_vector_x)
                radius_vector_y = self.shift_coords(radius_vector_y)
                radius_x = np.linalg.norm(radius_vector_x - centre)
                radius_y = np.linalg.norm(radius_vector_y - centre)
                radius = [int(radius_x), int(radius_y)]
                is_ellipse = True

        centre = (int(centre[0]),int(centre[1]))

        if not self.fill_colour is None:
            if is_ellipse:
                cv2.ellipse(layer, centre, radius, 0,0,360,color=self.fill_colour, thickness=-1)
            else:
                cv2.circle(layer, centre, radius, self.fill_colour, -1)

        if not self.stroke_colour is None:
            if is_ellipse:
                cv2.ellipse(layer, centre, radius, 0,0,360,color=self.stroke_colour, thickness=self.stroke_weight)
            else:
                cv2.circle(layer, centre, radius, self.stroke_colour, self.stroke_weight)

        if annotate:
            cv2.circle(layer, centre, 1, self.text_colour, -1)
            cv2.putText(
                img = layer,
                text = f"{centre[0]},{centre[1]}",
                org = centre,
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 0.3,
                color = self.text_colour,
                thickness = 1 )
            cv2.line(layer, centre, (centre[0], centre[1]-radius), self.text_colour,1)
            cv2.putText(
                img = layer,
                text = f"{radius}",
                org = (centre[0], centre[1]-radius),
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 0.3,
                color = self.text_colour,
                thickness = 1 )

    def line(self, pt1 = (0,0), pt2 = (100,100), layer = None, annotate=False):
        """
        Draw line
        
        Args:
            pt1 (np.array): Coordinates of start
            pt2 (int): Coordinates of end
            layer (np.array): Where to draw, defaults to dot.canvas
            annotate (bool): If true, dimensions are annotated on sketch (for debug)
        """
        if layer is None:
            layer = self.canvas

        if not self.current_transform is None:
            pt1 = self.shift_coords(pt1)
            pt2 = self.shift_coords(pt2)

        pt1 = (int(pt1[0]),int(pt1[1]))
        pt2 = (int(pt2[0]),int(pt2[1]))
        
        if not self.stroke_colour is None:
            cv2.line(layer, pt1, pt2, self.stroke_colour, self.stroke_weight)

        if annotate:
            cv2.putText(
                img = layer,
                text = f"{pt1[0]},{pt1[1]}",
                org = pt1,
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 0.3,
                color = self.text_colour,
                thickness = 1 )
            cv2.putText(
                img = layer,
                text = f"{pt2[0]},{pt2[1]}",
                org = pt2,
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 0.3,
                color = self.text_colour,
                thickness = 1 )
            
    def poly(self, pts = [(0,100),(0,200),(100,150)], layer = None, annotate = False):
        """
        Draw poly shape given array of points
        
        Args:
            pts (np.array): Coordinates of points to draw
            layer (np.array): Where to draw, defaults to dot.canvas
            annotate (bool): If true, dimensions are annotated on sketch (for debug)
        """
        if layer is None:
            layer = self.canvas

        pts = np.array(pts, np.int32)
         
        if not self.current_transform is None:
            pts = np.array([self.shift_coords(p) for p in pts], np.int32)

        if not self.fill_colour is None:
            cv2.fillPoly(layer, [pts], self.fill_colour)

        if not self.stroke_colour is None:
            cv2.polylines(layer, [pts], True, self.stroke_colour, self.stroke_weight)

        if annotate:
            for pt in pts:
                cv2.putText(
                    img = layer,
                    text = f"{pt[0]},{pt[1]}",
                    org = pt,
                    fontFace = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = 0.3,
                    color = self.text_colour,
                    thickness = 1 )
    
    def rectangle(self, pt1 = (0,0), pt2 = (100,100), layer = None, annotate = False):
        """
        Draw rectangle
        
        Args:
            pt1 (np.array): Coordinates of top left
            pt2 (int): Coordinates of bottom right
            layer (np.array): Where to draw, defaults to dot.canvas
            annotate (bool): If true, dimensions are annotated on sketch (for debug)
        """
        w = pt2[0] - pt1[0]
        h = pt2[1] - pt1[1]

        pts = np.array([pt1,[pt1[0]+w,pt1[1]],pt2,[pt1[0],pt1[1]+h]], np.int32)
         
        self.poly(pts, layer, annotate)

    def shift_coords(self, coords):
        coords = np.array(coords)
        coords = np.append(coords, 1)
        coords = np.dot(self.current_transform, coords)
        coords = coords[:2]
        return coords

    def rotate(self, theta, origin=None):
        """
        Rotate all shapes drawn after
        
        Args:
            theta (int): rotation in radians
            origin (np.array): point about which to rotate, defaults to (0,0)
        """
        if not origin is None:
            self.translate(origin)
        m = np.array([[np.cos(theta), -np.sin(theta),0],
                          [np.sin(theta), np.cos(theta),0],
                          [0,0,1]])
        self.apply_transform(m)
        if not origin is None:
            self.translate(origin*-1)
    
    def translate(self, origin):
        """
        Translate all shapes drawn after
        
        Args:
            origin (np.array): point to move origin to
        """
        m = np.array([[1, 0, origin[0]],
                    [0, 1, origin[1]],
                    [0, 0, 1]])
        self.apply_transform(m)
    
    def scale(self, sx=1, sy=1, origin=None):
        """
        Scale all shapes drawn after
        
        Args:
            sx (float): scale in x direction 
            sy (float): scale in y direction 
            origin (np.array): point about which to scale, defaults to (0,0)
        """
        if not origin is None:
            self.translate(origin)
        m = np.array([[sx, 0, 0],
                    [0, sy, 0],
                    [0, 0, 1]])
        self.apply_transform(m)
        if not origin is None:
            self.translate(origin*-1)

    def apply_transform(self, m):
        if not self.current_transform is None:
            self.current_transform = self.current_transform @ m 
        else:
            self.current_transform = m

    def reset_transforms(self):
        """
        Reset transformations (use to isolate transforms within draw loop)
        """
        self.current_transform = None

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
        #Border value is really important here, we replace with (1,1,1) which is the nothing pixel
        transformed_image = cv2.warpAffine(src, transformation_matrix[:2, :], (cols, rows), borderValue=(1,1,1))
        return transformed_image

    
    def transform_layer(self, layer, m, origin = (0,0)):
        """
        Transform a layer given the linear matrix (2,2) and an origin   

        Args:
            layer (np.array): The layer to scale
            m (np.array): A 2x2 matrix for the linear transform
            origin (tuple): The origin about which to make the transform

        Returns:
            np.array: The transformed version of the layer
        """
        return self.linear_transformation(layer, m, origin)
    
    def scale_layer(self, layer, sx=1, sy=1, origin =(0,0)):
        """
        Scale layer given x and y factors and an origin   

        Args:
            layer (np.array): The layer to scale
            sx (float): The scale factor in the x axis. Defaults to 1
            sy (float): The scale factor in the y axis. Defaults to 1
            origin (tuple): The origin about which to make the transform

        Returns:
            np.array: The transformed version of the layer
        """
        m = np.array([[sx,0.0],
                          [0.0,sy]])
        return self.transform_layer(layer, m, origin)
    
    def rotate_layer(self, layer, theta, origin = (0,0)):
        """
        Rotate layer given theta and an origin 

        Args:
            layer (np.array): The layer to scale
            theta (float): The rotation in radians
            origin (tuple): The origin about which to make the transform

        Returns:
            np.array: The transformed version of the layer
        """

        m = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
        return self.transform_layer(layer, m, origin)
    
    #Callback for mouse moved
    def mouse_moved(self, event, x, y, flags, param):
        self.mouse_x = x
        self.mouse_y = y
        # print(event)
        if event == 1:
            self.mouse_down = True
        elif event == 4:
            self.mouse_down = False

    def background(self, col=(0,0,0)):
        """
        Fill layer with given RGB colour

        Args:
            col (tuple): The RGB colour (unit8) to fill. Defaults to black.
        """
        cv2.rectangle(self.canvas, (0,0), (self.width,self.height), col, -1)

    def draw_playhead(self, layer, audio_output=0):
        if audio_output < len(self.music.audio_outputs):
            output = self.music.audio_outputs[audio_output]
            if isinstance(output, SamplePlayer):
                latency = output.buffer_size * output.audio_latency
                mixed = output.y.mean(axis=0)
                playhead = int(((output.current_sample-latency) / len(mixed)) *self.width)
                cv2.line(layer, (playhead , 0), (playhead, self.height), self.white, 5)

    def draw_waveform(self, layer, audio_output = 0, col=None, with_playhead = False):
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
                mixed = output.y.mean(axis=0)
                samples_per_pixel = len(mixed) / self.width
                for i in range(self.width):
                    val = mixed[int(samples_per_pixel*i)]
                    h = int(val * self.height)
                    y = self.height//2 - h//2
                    if not col == None:
                        cv2.line(layer, (i, y), (i, y + h), col, 2)
                if with_playhead:
                    self.draw_playhead(layer, audio_output)
                    
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
        to_paste = np.array(to_paste)
        x = coords[0]
        y = coords[1]
        w = to_paste.shape[1]
        h = to_paste.shape[0]
        cw = layer.shape[1]
        ch = layer.shape[0]
        if to_paste.ndim == 2:
            to_paste = to_paste[:,:,np.newaxis]
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
            upper_objects = cv2.bitwise_and(c2[0], c2[0], mask=upper_layer_objects_mask)
            #Dont blend into parts of lower layer where there isnt stuff in the upper layer
            lower_hidden_by_upper = cv2.bitwise_and(c1[0], c1[0], mask=upper_layer_objects_mask)
            #Blend appropriate bits
            c2[0] = (upper_objects*c2[1]) + (lower_hidden_by_upper*(1-c2[1]))
            upper_layer_no_objects_mask = cv2.bitwise_not(upper_layer_objects_mask)
            lower_not_hidden_by_upper = cv2.bitwise_and(c1[0], c1[0], mask=upper_layer_no_objects_mask)
            #Add in blended stuff (not over unblended stuff)
            c2[0] = np.array(c2[0] + lower_not_hidden_by_upper, dtype = np.uint8)
            c2[0][c2[0]>255] = 255
        self.canvas = self.layers[-1][0].copy()
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
    
    def start_record(self, audio_output=0, end=None):
        """
        Start collecting frames
        """
        if not self.recording:
            print("starting record", self.millis, end)
            self.video_recording_buffer = []
            self.start_record_time = self.millis
            if audio_output < len(self.music.audio_outputs):
                output = self.music.audio_outputs[audio_output]
                output.recording_buffer = []
                output.recording = True
            self.recording = True
            if not end == None:
                self.end_recording_at = self.millis + end
            
    
    def stop_record(self, output_video_path = "output.mp4", fps = 25, audio_output = 0, audio_latency = 6):
        """
        Stop collecting frames and render capture
        Args:
            output_video_path (str): where to save file
            fps (int): The frame rate to render the video. Defaults to 25
            audio_output (int): which audio device to use
            audio_latency (int): number of frames to pad to resync audio with video
        """
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
                    with wave.open(file_name, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)  # 16-bit audio
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes((audio_frames * 32767).astype(np.int16).tobytes())

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
                    audio_data = audio_data[:,:,0]
                    #padd some zeros to get back in sync with visuals (audio is early apparently)
                    audio_data = np.pad(audio_data, ((audio_latency,0), (0, 0)), mode='constant', constant_values=0)
                    save_audio_to_wav(audio_data, sample_rate, audio_file)
                    combine_audio_video(audio_file, output_video_path, combined_file)
                    output.audio_recording_buffer = []
                    output.recording = False
                else:
                    print("no audio to write to file")

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
        
        frame_target_length = (1/self.fps)*1000
        cv2.namedWindow(name)
        cv2.setMouseCallback(name,self.mouse_moved)
        try :
            done = False
            setup()
            self.start_time_millis = int(round(time.time() * 1000))
            while not done:
                frame_started_at = int(round(time.time() * 1000))
                draw()                
                self.update_canvas()
                #Draw to window
                canvas_rgb = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
                cv2.imshow(name, canvas_rgb)
                
                if self.recording:
                    self.video_recording_buffer.append({"frame":canvas_rgb,"timestamp":self.millis})
                
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

                self.current_transform = None

                frame_length = int(round(time.time() * 1000)) - frame_started_at
                if frame_length < frame_target_length:
                    sleep((frame_target_length - frame_length)/1000)
                
                if self.recording and self.end_recording_at < self.millis:
                    try:
                        self.stop_record()
                    except Exception as e:
                        print("error recording video")
                        print(e)
                        traceback.print_exc()
                    self.end_recording_at = np.inf

        except Exception as e:
            done = True
            print(e)
            traceback.print_exc()
            self.exit()            
        
        self.exit()