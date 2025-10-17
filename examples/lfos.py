from dorothy import Dorothy
import numpy as np

dot = Dorothy(640,480)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)     

    def setup(self):
        # Create LFOs for different parameters and save ids
        self.size_lfo = dot.get_lfo('sine', freq=0.5, range=(20, 100))
        self.x_lfo = dot.get_lfo('saw', freq=1.0, range=(0, dot.width))
        self.color_lfo = dot.get_lfo('triangle', freq=2.0, range=(0, 255))
        self.rotation_lfo = dot.get_lfo('sine', freq=0.3, range=(0, 2 * np.pi))

    def draw(self):
        dot.background(dot.black)
        
        # Use LFO values
        size = dot.lfo_value(self.size_lfo)
        x = dot.lfo_value(self.x_lfo)
        color_val = int(dot.lfo_value(self.color_lfo))
        rotation = dot.lfo_value(self.rotation_lfo)
        
        # Draw with modulated parameters
        dot.fill((color_val, 100, 255))
        with dot.transform():
            dot.translate(x, dot.height // 2)
            dot.rotate(rotation)
            dot.circle((0, 0), size)

MySketch()