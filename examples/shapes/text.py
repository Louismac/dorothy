"""Text rendering: static labels, live values, and text inside transforms."""
from dorothy import Dorothy

dot = Dorothy(640, 480)

class MySketch:

    def setup(self):
        pass

    def draw(self):
        dot.background(dot.black)

        # text(string, x, y, size)
        dot.fill(dot.white)
        dot.text("Hello Dorothy", 20, 50, 32)
        dot.text(f"frame {dot.frames}   mouse ({dot.mouse_x}, {dot.mouse_y})", 20, 100, 18)

        # Text inside a transform rotates with it
        with dot.transform():
            dot.translate(dot.width // 2, dot.height // 2)
            dot.rotate(dot.frames * 0.02)
            dot.fill(dot.yellow)
            dot.text("rotating text", -80, 0, 24)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
