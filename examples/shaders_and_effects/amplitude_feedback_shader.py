"""Feedback shader: each frame zooms slightly into the previous frame.

apply_shader() with bake=True writes back into the canvas, so shapes
drawn this frame are composited on top of the zoomed prior frame.
"""
from dorothy import Dorothy

dot = Dorothy(640, 480)

class MySketch:

    def setup(self):
        dot.music.start_file_stream("../audio/gospel.wav", fft_size=512)
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
        dot.set_stroke_weight(5)
        dot.stroke(dot.blue)
        dot.circle((dot.width // 2, dot.height // 2), 100)
        dot.set_stroke_weight(20)
        dot.rectangle((dot.width // 2, dot.height // 2), (100, 100))
        dot.set_stroke_weight(10)
        dot.line((dot.width // 2, dot.height // 2), (0, 0))
        # zoom < 1.0 contracts toward centre; amplitude makes it pulse
        dot.apply_shader(self.feedback, bake=True,
                         zoom=1 - dot.music.amplitude() * 0.25)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
