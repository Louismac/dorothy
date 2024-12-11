import numpy as np
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup)

    def setup(self):
        self.phase = 0.0 
        sr = 44100

        #Audio Callback function
        def get_frame(size):
            #Get parameters from mouse
            frequency = dot.mouse_x
            amplitude = dot.mouse_y/dot.height
            #Get increments
            delta = 2 * np.pi * frequency / sr 
            x = delta * np.arange(size)
            #Make some sound
            audio = amplitude * np.sin(self.phase + x)
            #update phase
            self.phase += delta * size 
            return audio
        
        dot.music.start_dsp_stream(get_frame, sr = sr)

MySketch()          







