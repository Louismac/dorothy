from dorothy import Dorothy
from PIL import Image

dot = Dorothy(640,480)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    def setup(self):
        #Listen to mic or internal loop back (e.g. blackhole)
        dot.music.start_device_stream(2)
        self.pixelate = '''
        #version 330
        uniform sampler2D texture0;
        uniform vec2 resolution;
        uniform float pixelSize;
        in vec2 v_texcoord;
        out vec4 fragColor;

        void main() {
            vec2 pixels = resolution / pixelSize;
            vec2 uv = floor(v_texcoord * pixels) / pixels;
            fragColor = texture(texture0, uv);
        }
        '''
        self.layer = dot.get_layer()
        self.rgb_image = Image.open('../images/space.jpg')
        
    def draw(self):
        dot.paste(self.rgb_image, (0, 0),(dot.width, dot.height))
        with dot.layer(self.layer):
            dot.background(dot.white)
            dot.fill(dot.red)
            dot.circle((dot.width//2,dot.height//2),int(dot.music.amplitude()*dot.height*10))
            #Remember to accumulate changes through the chain
            dot.pixelate(pixel_size=8.0, accumulate=True)
            dot.roll(offset_x=dot.frames, accumulate=True)
            dot.tile(8,8, accumulate=True)
            dot.cutout(dot.red)
        dot.draw_layer(self.layer)

MySketch()   
    







