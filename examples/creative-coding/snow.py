from dorothy import Dorothy 
import numpy as np

dot = Dorothy() 
class MySketch:
    
    def __init__(self):
        
        dot.start_loop(self.setup, self.draw)
        
    def setup(self):
        num_flakes = 100
        x = np.linspace(0,dot.width,num_flakes).astype(int)
        #random ys stop grouping
        y = np.random.randint(0, dot.height, num_flakes)
        self.speed = np.random.randint(1, 7, 100)
        self.snow = []
        #iterate through both and combine into the variable pt
        for pt in zip(x,y):
            self.snow.append(list(pt))

    def draw(self):
        ptr = 0
        dot.fill(dot.white)
        for pt in self.snow:
            speed = self.speed[ptr]
            dot.circle(pt, speed)
            self.snow[ptr][1] = (self.snow[ptr][1] + speed) % dot.height
            ptr = ptr + 1 
        #semi transparent layer to overwrite 
        cover = dot.get_layer()
        dot.fill(dot.darkblue)
        dot.rectangle((0,0),(dot.width, dot.height),layer=cover)
        #The higher the alpha, the more opaque the cover, so small alpha makes longer trails 
        dot.draw_layer(cover, 0.2)     
MySketch() 