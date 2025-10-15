from dorothy import Dorothy 

dot = Dorothy()

class MySketch:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        print(dot.width,dot.height)
        dot.background((200,120,0))
    
    def draw(self):
        
        dot.fill(dot.cyan)
        dot.circle((dot.frames%dot.width, 300), 100)

        dot.fill(dot.darkgreen)
        dot.circle(((dot.frames*2)%dot.width, (dot.frames*2)%dot.width), 100)

        # dot.fill((200,120,0,10))
        # dot.rectangle((0,0),(dot.width,dot.height))

MySketch()