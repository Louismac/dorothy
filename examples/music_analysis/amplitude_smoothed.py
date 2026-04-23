"""Smooth amplitude with a sliding-window average to reduce jitter."""
from dorothy import Dorothy

dot = Dorothy(640, 480)

class MySketch:

    def setup(self):
        dot.music.start_file_stream("../audio/disco.wav")
        # get_window(n) keeps the last n values and returns their mean
        self.mean_amp = dot.get_window(20)

    def draw(self):
        r = self.mean_amp.add(dot.music.amplitude())
        dot.background(dot.white)
        dot.fill(dot.red)
        dot.circle((dot.width // 2, dot.height // 2), 1 + int(r * dot.height))

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
