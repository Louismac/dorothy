#RUN livecode.py FIRST!
#Changes you make and then save this file will be reflected in the window

class MySketch:

    def setup(self, dot):
        self.counter = 1

    def run_once(self, dot):
        dot.background(dot.black)
        dot.no_fill()
        dot.stroke(dot.white)

    def draw(self, dot):
        dot.background(dot.black)
        dot.rectangle((100,100),(dot.mouse_x,dot.mouse_y),annotate=True)

        # dot.fill(dot.red)
        # dot.stroke(dot.blue)
        # dot.set_stroke_weight(self.counter%100)
        # dot.circle((dot.frame%dot.width,dot.height//2), 100)