from dorothy import Dorothy

dot = Dorothy(640,480)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    def setup(self):
        #Listen to mic or internal loop back (e.g. blackhole)
        dot.music.start_device_stream(1)
        self.feedback = '''
        #version 330
        uniform sampler2D texture0;
        uniform float zoom;
        in vec2 v_texcoord;
        out vec4 fragColor;
        
        void main() {
            vec2 uv = (v_texcoord - 0.5) * zoom + 0.5;
            vec4 color = texture(texture0, uv);
            color.rgb *= 0.98;
            fragColor = color;
        }
        '''
        
    def draw(self):
        dot.fill(dot.red)
        dot.circle((dot.width//2,dot.height//2),100)
        dot.apply_shader(self.feedback, accumulate=True, zoom=1-(dot.music.amplitude()*5))

MySketch()   
    







