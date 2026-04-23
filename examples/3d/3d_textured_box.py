"""Box with a different live-drawn layer on each face."""
from dorothy import Dorothy
from PIL import Image

dot = Dorothy(640, 480)

class MySketch:

    def setup(self):
        dot.camera_3d()
        dot.set_camera((10, 10, 10), (0, 0, 0))
        # One layer per face; pass as a dict to dot.box()
        self.faces = {k: dot.get_layer()
                      for k in ('front', 'back', 'right', 'left', 'top', 'bottom')}
        self.photo = Image.open('../images/space.jpg')

    def draw(self):
        dot.camera_2d()
        with dot.layer(self.faces['front']):
            dot.paste(self.photo, (0, 0))

        colors = dict(back=dot.blue, right=dot.yellow, left=dot.red,
                      top=dot.magenta, bottom=dot.cyan)
        for i, (name, color) in enumerate(colors.items()):
            with dot.layer(self.faces[name]):
                dot.fill(color)
                dot.circle((dot.frames % dot.width, dot.frames % dot.height), 20 + i * 5)

        dot.background((10, 10, 20))
        dot.camera_3d()
        with dot.transform():
            dot.rotate(dot.frames * 0.01,  0, 1, 0)
            dot.rotate(dot.frames * 0.005, 1, 0, 0)
            dot.box((2, 2, 2), texture_layers=self.faces)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
