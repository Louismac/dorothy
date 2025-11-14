from dorothy import Dorothy

dot = Dorothy(640,480)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    def setup(self):
        file_path = "../audio/gospel.wav"
        dot.music.start_file_stream(file_path, fft_size=512)
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
        
    def draw(self):
        dot.background(dot.white)
        dot.fill(dot.red)
        dot.circle((dot.width//2,dot.height//2),int(dot.music.amplitude()*dot.height*2))
        dot.apply_shader(self.pixelate, pixelSize=8.0, accumulate=False)

MySketch()   
    







