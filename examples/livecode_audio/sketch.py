#RUN livecode.py FIRST!
#Changes you make and then save this file will be reflected in the window
import numpy as np
import librosa

class MySketch:

    def setup(self, dot):
        self.phase = 0.0 
        self.sr = 44100

        def get_frame(size):
            return np.zeros(size)
        index = dot.music.start_dsp_stream(get_frame, sr = self.sr)

        # analyse chroma when a new buffer is retrieved
        self.chroma = np.zeros(12)
        def on_new_frame(buffer=np.zeros(2048)):
            c = librosa.feature.chroma_stft(y=buffer, n_fft=512)
            self.chroma = c.mean(axis=1)
        dot.music.audio_outputs[index].on_new_frame = on_new_frame
        dot.stroke(dot.red)
        dot.set_stroke_weight(10)

    def draw(self, dot):
        #Audio Callback function
        def get_frame(size):
            #Get parameters from mouse
            frequency = 440
            amplitude = dot.mouse_y/dot.height
            #Get increments
            delta = 2 * np.pi * frequency / self.sr 
            x = delta * np.arange(size)
            #Make some sound
            audio = amplitude * np.sin(self.phase + x)
            #update phase
            self.phase += delta * size 
            return audio
        #update audio callback
        dot.music.audio_outputs[0].get_frame = get_frame

        #draw chromagram
        dot.background(dot.white)
        for i,c in enumerate(self.chroma):
            x = (dot.width//12)*i
            y = int(dot.height - (dot.height*c))
            dot.line((x, dot.height), (x, y))