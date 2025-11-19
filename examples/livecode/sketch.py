from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  
            
    def setup(self):
        self.col = (0,255,0)
        print("start")

    def run_once(self):
        print("run once")
        self.col = (255,0,255)
                
    def draw(self):
        dot.background(self.col)
        dot.fill(dot.red)
        dot.rectangle((0,dot.frames%40),(100,100))



