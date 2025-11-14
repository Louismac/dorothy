from dorothy import Dorothy 

dot = Dorothy()

class MySketch:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        self.layer = dot.get_layer()
        
    def draw(self):
        #redo background on main canvas
        dot.background((255,0,255))
        #draw trails to layer
        with dot.layer(self.layer):
            dot.background((255,0,255,10))
            dot.fill(dot.yellow)
            dot.circle((dot.mouse_x, dot.mouse_y), 100)

        dot.draw_layer(self.layer)

        

MySketch()