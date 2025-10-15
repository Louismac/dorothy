from dorothy import Dorothy 
import moderngl

dot = Dorothy()

class MySketch:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        dot.background((200,120,0))
    
    def draw(self):
        
        dot.fill((200, 120, 0, 5))
        dot.rectangle((0, 0), (dot.width, dot.height))
                
        dot.fill(dot.cyan)
        dot.circle((dot.frames % dot.width, 300), 100)

        

MySketch()