"""Draw static content into a layer once, then rotate the layer every frame."""
from dorothy import Dorothy
import numpy as np

dot = Dorothy()

class MySketch:

    def setup(self):
        dot.music.start_file_stream("../audio/disco.wav", buffer_size=2048)
        self.layer = dot.get_layer()
        # Draw once into the layer; no need to redraw each frame
        with dot.layer(self.layer):
            dot.fill(dot.yellow)
            dot.rectangle((dot.width // 4, dot.height // 4),
                           (dot.width * 3 // 4, dot.height * 3 // 4))
        self.theta = 0.0

    def draw(self):
        # Low-alpha fill creates a motion trail instead of clearing
        dot.fill((255, 0, 0, 20))
        dot.rectangle((0, 0), (dot.width, dot.height))
        self.theta += dot.music.amplitude() * np.pi
        cx, cy = dot.width // 2, dot.height // 2
        with dot.transform():
            dot.translate(cx, cy)
            dot.rotate(self.theta)
            dot.translate(-cx, -cy)
            dot.draw_layer(self.layer)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
