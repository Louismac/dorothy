from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        
    def draw(self):
        dot.stroke((255, 0, 0))
        dot.no_fill()
        # Try different weights
        for i in range(1, 11):
            dot.set_stroke_weight(i)
            dot.rectangle((300, 50 + i*20), (750, 50 + i*60))
            dot.circle((50+ i*20, 50 + i*20), (100))
            dot.line((50, 50 + i*20), (250, 50 + i*20))
            

MySketch()          