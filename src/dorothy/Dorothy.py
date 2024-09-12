import numpy as np
import cv2
from cv2 import rectangle, line
import signal
import sys
import numpy as np
import time
import traceback
import glob
from .css_colours import css_colours
from .Audio import *

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
        #Border value is really important here, we replace with (1,1,1) which is the nothing pixel
        transformed_image = cv2.warpAffine(src, transformation_matrix[:2, :], (cols, rows), borderValue=(1,1,1))
        return transformed_image

    
    def transform(self, layer, m, origin = (0,0)):
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
    
    def scale(self, layer, sx=1, sy=1, origin =(0,0)):
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
        return self.transform(layer, m, origin)
    
    def rotate(self, layer, theta, origin = (0,0)):
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
        return self.transform(layer, m, origin)
    
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
        Fill layer with given RGB colour

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
            #Swap the 1s for 0s so we dont overflow
            c2[0][c2[0]==1] = 0
            upper_objects = cv2.bitwise_and(c2[0], c2[0], mask=upper_layer_objects_mask)
            #Dont blend into parts of lower layer where there isnt stuff in the upper layer
            lower_hidden_by_upper = cv2.bitwise_and(c1[0], c1[0], mask=upper_layer_objects_mask)
            #Blend appropriate bits
            c2[0] = (upper_objects*c2[1]) + (lower_hidden_by_upper*(1-c2[1]))
            upper_layer_no_objects_mask = cv2.bitwise_not(upper_layer_objects_mask)
            lower_not_hidden_by_upper = cv2.bitwise_and(c1[0], c1[0], mask=upper_layer_no_objects_mask)
            #Add in blended stuff (not over unblended stuff)
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