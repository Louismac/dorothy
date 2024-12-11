#Changes you make and then save this file will be reflected in the window

class MySketch:

    def setup(self, dot):
        self.counter = 1

    def draw(self, dot):
        self.counter += 1
        dot.fill(dot.green)
        dot.stroke(dot.blue)
        dot.set_stroke_weight(5)
        dot.circle((200,200), 100)