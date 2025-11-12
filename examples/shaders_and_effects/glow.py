from dorothy import Dorothy

dot = Dorothy(800,800)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        self.emit_shader = '''
        #version 330
        uniform sampler2D texture0;
        uniform float glow_boost;
        in vec2 v_texcoord;
        out vec4 fragColor;
        
        void main() {
            vec4 color = texture(texture0, v_texcoord);
            
            // Boost bright colors
            float brightness = length(color.rgb);
            if (brightness > 0.5) {
                color.rgb *= (1.0 + glow_boost);
            }
            
            fragColor = color;
        }
        '''
        
        self.fast_blur = '''
        #version 330
        uniform sampler2D texture0;
        uniform vec2 resolution;
        in vec2 v_texcoord;
        out vec4 fragColor;
        
        void main() {
            vec2 pixel = 1.0 / resolution;
            vec4 color = vec4(0.0);
            
            // 3x3 blur
            for(int x = -1; x <= 1; x++) {
                for(int y = -1; y <= 1; y++) {
                    color += texture(texture0, v_texcoord + vec2(x, y) * pixel);
                }
            }
            
            fragColor = color / 9.0;
        }
        '''

    def draw(self):
        # Draw bright objects
        dot.background((0,0,0,20))
        dot.fill((0, 100, 0))
        dot.circle((dot.mouse_x, dot.mouse_y), 50)
        
        # Boost emission
        dot.apply_shader(self.emit_shader, accumulate=True, glow_boost=0.99)
        
        # Blur multiple times for softer glow
        for _ in range(10):
            dot.apply_shader(self.fast_blur, accumulate=True)

MySketch()