#RUN livecode.py FIRST!
#Changes you make and then save this file will be reflected in the window

class MySketch:

    def setup(self, dot):
        self.counter = 1

    def draw(self, dot):
        dot.background(0)
        self.counter += 1
        dot.fill(dot.green)
        dot.stroke(dot.blue)
        dot.set_stroke_weight(self.counter%100)
        dot.circle((dot.width//2,dot.height//2), 200)