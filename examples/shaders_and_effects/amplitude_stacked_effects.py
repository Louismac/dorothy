from dorothy import Dorothy
from PIL import Image

dot = Dorothy(800,800)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    def setup(self):
        #Listen to mic or internal loop back (e.g. blackhole)
        dot.music.start_device_stream(2)
        self.layer = dot.get_layer()
        self.rgb_image = Image.open('../images/space.jpg')
        self.lfo = dot.get_lfo("sine", 0.01, (1,8))
        
    def draw(self):
        
        dot.background(dot.black)

        with dot.layer(self.layer):
            dot.background(dot.black)
            dot.fill(dot.white)
            dot.circle((dot.width//2,dot.height//2),200)
            #Remember to accumulate changes through the chain
            dot.roll(offset_x=dot.frames*4, accumulate=True)
            dot.tile(8,8, accumulate=True)
            dot.cutout(dot.white, accumulate=True)


        with dot.transform():
            dot.translate(dot.centre[0], dot.centre[1])
            dot.scale(1+dot.music.amplitude()*20)
            dot.translate(-dot.centre[0], -dot.centre[1])
            dot.paste(self.rgb_image, (0, 0),(dot.width, dot.height))
            dot.draw_layer(self.layer)
        
        tile = dot.lfo_value(self.lfo)
        dot.tile(tile,tile, accumulate=True)

MySketch()   
    







