"""Mouse position and press/release events."""
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def setup(self):
        self.color = dot.red

        def on_press(x, y, button):
            print(f"button {button} pressed at ({x}, {y})")
            self.color = dot.blue if self.color == dot.red else dot.red

        dot.on_mouse_press = on_press

    def draw(self):
        dot.fill(self.color)
        # dot.mouse_x / dot.mouse_y update every frame automatically
        dot.circle((dot.mouse_x, dot.mouse_y), 100)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
