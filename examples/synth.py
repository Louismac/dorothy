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
            phase_increment = 2 * np.pi * frequency / sr 
            x = phase_increment * np.arange(size)
            #Make some sound
            audio = amplitude * np.sin(self.phase + x)
            #update phase
            self.phase += phase_increment * size 
            return audio
        
        dot.music.start_dsp_stream(get_frame, sr = sr)

MySketch()          







