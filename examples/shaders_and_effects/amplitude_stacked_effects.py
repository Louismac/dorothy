"""Stack multiple layer effects: tile, roll, cutout, then composite."""
from dorothy import Dorothy
from PIL import Image
import numpy as np

dot = Dorothy(800, 800)

class MySketch:

    def setup(self):
        dot.music.start_file_stream("../audio/gospel.wav", fft_size=512)
        self.mask        = dot.get_layer()
        self.trail_layer = dot.get_layer()
        self.space_layer = dot.get_layer()
        self.rgb_image   = Image.open('../images/space.jpg')
        self.mean_amp    = dot.get_window(30)

    def draw(self):
        dot.background(dot.black)
        amp = self.mean_amp.add(dot.music.amplitude())

        with dot.layer(self.mask):
            dot.background(dot.black)
            dot.fill(dot.white)
            dot.circle((dot.width // 2, dot.height // 2), 300)
            # bake=True writes each effect back so operations compound
            dot.tile(5, 5, bake=True)
            dot.roll(offset_x=amp * -100, offset_y=amp * 100, bake=True)
            dot.cutout(dot.white, bake=True)

        with dot.layer(self.trail_layer):
            with dot.transform():
                dot.translate(dot.centre[0], dot.centre[1])
                dot.scale(1 + amp * 3)
                dot.rotate(np.pi * (dot.frames / 1000))
                dot.translate(-dot.centre[0], -dot.centre[1])
                dot.background((0, 0, 0, 5))
                dot.draw_layer(self.mask, 0.8)

        with dot.layer(self.space_layer):
            dot.paste(self.rgb_image, (0, 0), (dot.width, dot.height))
            dot.roll(offset_x=amp * 40 + dot.frames * 2, bake=True)

        dot.draw_layer(self.space_layer)
        dot.draw_layer(self.trail_layer)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
