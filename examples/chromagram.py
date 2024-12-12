from dorothy import Dorothy 
import librosa
import numpy as np
from cv2 import line
from utils import sine_step

dot = Dorothy(1200,300) 
class MySketch:
    def __init__(self):
        dot.start_loop(self.setup, self.draw) 

    def setup(self):

        audio = sine_step()
        id = dot.music.start_sample_stream(audio, sr=22050, buffer_size = 2048)

        self.chroma = np.zeros(12)
        def on_new_frame(buffer=np.zeros(2048)):
            try:
                self.chroma = librosa.feature.chroma_stft(y=buffer, n_fft=512)[0,...].mean(axis=1)
            except:
                print("error")
            
        dot.music.audio_outputs[id].on_new_frame = on_new_frame

    def draw(self):
        dot.background(dot.white)
        for i,c in enumerate(self.chroma):
            x = (dot.width//12)*i
            y = int(dot.height - (dot.height*c))
            line(dot.canvas, (x, dot.height), (x, y), dot.red, 10)
    
               
MySketch() 