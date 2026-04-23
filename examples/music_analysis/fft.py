"""FFT spectrum visualiser — vertical bars for each frequency bin."""
from dorothy import Dorothy

dot = Dorothy(640, 480)

class MySketch:

    def setup(self):
        dot.music.start_file_stream("../audio/gospel.wav", fft_size=512, buffer_size=512)

    def draw(self):
        dot.background(dot.black)
        # Sample every 8th bin; fft() returns normalised magnitudes (0–1)
        bins  = dot.music.fft()[::8]
        bar_w = dot.width // len(bins)
        for i, val in enumerate(bins):
            h = min(int(val * dot.height * 10), dot.height)
            x = i * bar_w + bar_w // 2
            dot.stroke((0, int((1 - val) * 255), 0))
            dot.set_stroke_weight(max(1, int(val * 3)))
            dot.line((x, dot.height), (x, dot.height - h))

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
