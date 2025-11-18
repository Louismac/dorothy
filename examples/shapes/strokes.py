from dorothy import Dorothy

dot = Dorothy(800,800)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        
    def draw(self):
        dot.stroke((255, 0, 0))
        dot.background(dot.black)
        # dot.no_fill()
        # Try different weights
        for i in range(1, 2):
            dot.set_stroke_weight(3)
            dot.rectangle((50, 51), (dot.mouse_x, dot.mouse_y))
            # dot.circle((50+ i*20, 50 + i*20), (100))
            # dot.line((50, 50 + i*20), (250, 50 + i*20))
            

MySketch()          