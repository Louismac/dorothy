"""Motion trails using a layer with a semi-transparent background.

Drawing with alpha < 255 to a layer each frame fades old content slowly,
creating a trail. dot.draw_layer() composites it onto the main canvas.
"""
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def setup(self):
        self.layer = dot.get_layer()

    def draw(self):
        dot.background((255, 0, 255))
        with dot.layer(self.layer):
            # Low alpha background erases previous circles gradually
            dot.background((255, 0, 255, 10))
            dot.fill(dot.yellow)
            dot.circle((dot.mouse_x, dot.mouse_y), 100)
        dot.draw_layer(self.layer)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
