from dorothy import Dorothy

dot = Dorothy(640,640)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        pass
            
    def draw(self):
        dot.background(dot.black)
        dot.renderer.render_text("Hello World", 50, 50, font_size=32, color=(1, 0, 0, 1))
        #dot.renderer.test_single_glyph()

MySketch()          







