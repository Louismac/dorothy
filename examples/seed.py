from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        
    def draw(self):
        dot.fill(dot.black)
        dot.stroke(dot.red)
        dot.set_stroke_weight(5)
        dot.circle((200,200), 100, annotate=True)
        dot.no_stroke()
        dot.fill(dot.green)
        dot.circle((20,200), 100, annotate=True)
        dot.no_fill()
        dot.stroke(dot.blue)
        dot.circle((20,400), 100, annotate=True)

MySketch()          