#RUN livecode.py FIRST!
#Changes you make and then save this file will be reflected in the window
import numpy as np

class MySketch:

    def setup(self, dot):
        self.phase = 0.0 
        self.sr = 44100
        def get_frame(size):
            return np.zeros(size)
        dot.music.start_dsp_stream(get_frame, sr = self.sr)
    
    def draw(self, dot):
        #Audio Callback function
        def get_frame(size):
            #Get parameters from mouse
            frequency = 100
            amplitude = dot.mouse_y/dot.height
            #Get increments
            delta = 2 * np.pi * frequency / self.sr 
            x = delta * np.arange(size)
            #Make some sound
            audio = amplitude * np.sin(self.phase + x)
            #update phase
            self.phase += delta * size 
            return audio
        
        dot.music.audio_outputs[0].get_frame = get_frame