from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    #Start loop in init
    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        self.col = dot.white

    def draw(self): 
        dot.background(self.col)
        dot.fill(dot.blue)
        dot.rectangle((0,dot.frames%40),(400,100)) 
               
#Initiate class
MySketch()          