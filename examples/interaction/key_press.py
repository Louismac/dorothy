import numpy as np
from cv2 import circle
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        self.color = dot.red

        def key_press(key, action, modifiers):
            if action == dot.keys.ACTION_PRESS:
                if key == dot.keys.SPACE:
                    print("SPACE key was pressed")
                if key == dot.keys.Z and modifiers.shift:
                    print("Shift + Z was pressed")

                if key == dot.keys.Z and modifiers.ctrl:
                    print("ctrl + Z was pressed")
            elif action == dot.keys.ACTION_RELEASE:
                if key == dot.keys.SPACE:
                    print("SPACE key was released")

            if self.color == dot.red:
                self.color = dot.blue
            else:
                self.color = dot.red

        dot.on_key_press = key_press

    def draw(self):
        dot.fill(self.color)
        dot.circle((dot.mouse_x, dot.mouse_y), 100)

MySketch()          