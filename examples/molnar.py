import numpy as np
import sounddevice as sd
from dorothy import Dorothy

dot = Dorothy()

class MySketch:
    
    show_beat = False
    beat_ptr = 0

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        #Play file from your computer
        file_path = "../audio/disco.wav"
        dot.music.start_file_stream(file_path, fft_size=512)
        self.pattern_layer = dot.get_layer()
        self.base_pattern()
        dot.background(dot.beige)
        
    def draw(self):
  
            factor = dot.music.amplitude() * 15 
            centre = np.array([dot.width//2, dot.height//2])
            dot.push_matrix()
            dot.translate(centre[0], centre[1])
            dot.scale(factor)
            dot.translate(-centre[0], -centre[1])
            dot.draw_layer(self.pattern_layer)
            dot.pop_matrix()
    
    #Draw the vera molnar grid to the pattern_layer (this gets transformed later)
    def base_pattern(self):
        
        dot.begin_layer(self.pattern_layer)
        dot.stroke((255, 37, 21))
        dot.set_stroke_weight(4)
        size = 30
        dot.background((0,0,0,0))
        for i in range(dot.width//size):
            for j in range(dot.height//size):
                y1 = j*size
                if np.random.random()<0.5:
                    y1 = (j+1)*size
                y2 = j*size
                if np.random.random()<0.5:
                    y2 = (j+1)*size
                dot.line((i*size,y1), ((i+1)*size,y2)) 
        dot.end_layer()
        
MySketch()         







