"""Keyboard events: press, release, and modifier keys."""
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def setup(self):
        self.color = dot.red

        def on_key(key, action, modifiers):
            # dot.keys constants mirror moderngl-window key names
            if action == dot.keys.ACTION_PRESS:
                if key == dot.keys.SPACE:
                    self.color = dot.blue if self.color == dot.red else dot.red
                if key == dot.keys.Z and modifiers.shift:
                    self.color = dot.green
                if key == dot.keys.Z and modifiers.ctrl:
                    self.color = dot.yellow
            elif action == dot.keys.ACTION_RELEASE:
                if key == dot.keys.SPACE:
                    print("SPACE released")

        dot.on_key_press = on_key

    def draw(self):
        dot.fill(self.color)
        dot.circle((dot.mouse_x, dot.mouse_y), 100)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
