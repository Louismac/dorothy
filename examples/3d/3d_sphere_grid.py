from dorothy import Dorothy
import math
dot = Dorothy(1080,960)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    def setup(self):
        #Listen to mic or internal loop back (e.g. blackhole)
        dot.music.start_device_stream(2)
        self.rgb_split = '''
        #version 330
        uniform sampler2D texture0;
        uniform float offset;
        in vec2 v_texcoord;
        out vec4 fragColor;

        void main() {
            float r = texture(texture0, v_texcoord + vec2(offset, 0.0)).r;
            float g = texture(texture0, v_texcoord).g;
            float b = texture(texture0, v_texcoord - vec2(offset, 0.0)).b;
            fragColor = vec4(r, g, b, 1.0);
        }
        '''
        dot.camera_3d()
        self.theta = 0
        dot.start_record(end = 10000)

    def draw(self):
        dot.background(dot.white)
        dot.fill(dot.red)
        amp = dot.music.amplitude()
        self.theta += amp
        x = 7 * math.cos(self.theta)
        z = 7 * math.sin(self.theta)
        dot.set_camera((x, 3, z), (0, 0, 0))
        
        # Grid centered at origin
        for i in range(10):
            for j in range(10):
                x = (i - 5) * 1  # -250 to 250
                y = (j - 5) * 1
                z = 0
                
                with dot.transform():
                    dot.translate(x, y, z)
                    dot.sphere(amp * 20)
                # print(x,y)
        dot.apply_shader(self.rgb_split, accumulate=False, offset=amp*0.5)

MySketch()   
    







