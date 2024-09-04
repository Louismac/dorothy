from cv2 import line
from src.Dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        
    def draw(self):
        line(dot.canvas, (0,0),(dot.frame%dot.width, dot.height), dot.red, 1)
        return

MySketch()          