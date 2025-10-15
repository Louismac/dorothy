from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  
            
    def setup(self):
        pass

    def run_once(self):
        dot.background(dot.red)
                
    def draw(self):
        dot.fill(dot.red)



