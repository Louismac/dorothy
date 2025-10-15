from dorothy import Dorothy

dot = Dorothy(640,480)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    # def setup(self):
    #     print("SETUP: drawing yellow circle")
    #     dot.background(dot.beige)
    #     dot.fill((255, 255, 0))
    #     dot.circle((200, 300), 50)

    # def draw(self):
    #     dot.fill((0, 255, 0))
    #     dot.circle((dot.frames, 300), 30)

    def setup(self):

        # Persistent canvas gets beige background
        dot.background(dot.beige)
        
        # Layer 0 has TRANSPARENT background with just yellow circle
        self.l = dot.get_layer()
        dot.begin_layer(self.l)
        # NO background call here - keep it transparent
        dot.fill((255, 255, 0))
        dot.circle((200, 300), 50)
        dot.end_layer()

    def draw(self):
        # Draw transparent layer (won't cover green circles)
        dot.draw_layer(self.l)
        dot.fill((0, 255, 0))
        dot.circle((dot.frames * 10, 300), 30)

MySketch()   
    







