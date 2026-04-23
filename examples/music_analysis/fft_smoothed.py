"""FFT spectrum with a sliding-window average for smoother animation."""
from dorothy import Dorothy

dot = Dorothy(640, 480)

class MySketch:

    def setup(self):
        self.skip = 8
        fft_size  = 512
        dot.music.start_file_stream("../audio/gospel.wav", fft_size=fft_size)
        # Multi-dimensional window: one running mean per frequency bin
        self.window = dot.get_window(20, dims=fft_size // self.skip)

    def draw(self):
        dot.background(dot.black)
        smoothed = self.window.add(dot.music.fft()[::self.skip])
        bar_w = dot.width // len(smoothed)
        for i, val in enumerate(smoothed):
            h = min(int(val * dot.height * 10), dot.height)
            x = i * bar_w + bar_w // 2
            dot.stroke((0, int((1 - val) * 255), 0))
            dot.set_stroke_weight(max(1, int(val * 3)))
            dot.line((x, dot.height), (x, dot.height - h))

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
