from dorothy import Dorothy
from PIL import Image
import numpy as np

dot = Dorothy(800,800)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    def setup(self):
        #Listen to mic or internal loop back (e.g. blackhole)
        file_path = "../audio/gospel.wav"
        dot.music.start_file_stream(file_path, fft_size=512)
        self.mask = dot.get_layer()
        self.trail_layer = dot.get_layer()
        self.space_layer = dot.get_layer()
        self.rgb_image = Image.open('../images/space.jpg')
        self.lfo = dot.get_lfo("sine", 0.1, (1,8))
        self.mean_amp = dot.get_window(30)
        
    def draw(self):
        
        #redo background on main canvas
        dot.background(dot.black)
        amp = self.mean_amp.add(dot.music.amplitude())

        with dot.layer(self.mask):
            dot.background(dot.black)
            dot.fill(dot.white)
            dot.circle((dot.width//2,dot.height//2), 300)
            #Remember to accumulate changes through the chain
            dot.tile(5,5, accumulate=True)
            dot.roll(offset_x=amp*-100,offset_y=amp*100,accumulate=True)
            dot.cutout(dot.white, accumulate=True)

        with dot.layer(self.trail_layer):
            with dot.transform():
                dot.translate(dot.centre[0], dot.centre[1])
                dot.scale(1+amp*3)
                dot.rotate(np.pi*(dot.frames/1000))
                dot.translate(-dot.centre[0], -dot.centre[1])
                dot.background((0,0,0,10))
                dot.draw_layer(self.mask)

        with dot.layer(self.space_layer):
            dot.paste(self.rgb_image, (0, 0),(dot.width, dot.height))
            dot.roll(offset_x=amp*40 + dot.frames*2, accumulate=True)

        dot.draw_layer(self.space_layer)
        dot.draw_layer(self.trail_layer)

MySketch()   
    







