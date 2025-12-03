import numpy as np
import librosa
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def setup(self):
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

    def draw(self):
        #Audio Callback function
        def get_frame(size):
            #Get parameters from mouse
            frequency = dot.mouse_x
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

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)