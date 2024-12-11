#RUN livecode.py FIRST!
#Changes you make and then save this file will be reflected in the window

class MySketch:

    def setup(self, dot):
        self.counter = 1

    def run_once(self, dot):
        dot.background(dot.purple)

    def draw(self, dot):
        dot.fill((dot.mouse_x,0,0))
        dot.stroke(dot.blue)
        dot.set_stroke_weight(self.counter%100)
        dot.circle((dot.width//2,dot.height//2), dot.frame%dot.width)