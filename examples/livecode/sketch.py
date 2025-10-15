from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  
            
    def setup(self):
        pass
                
    def draw(self):
        dot.background(dot.red)
        dot.fill(dot.green)
        dot.rectangle((0,dot.frames%400),(100,100))



