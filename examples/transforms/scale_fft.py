"""Scale bar heights by FFT frequency bin amplitudes."""
from dorothy import Dorothy

dot = Dorothy(640, 480)

class MySketch:

    def setup(self):
        dot.music.start_file_stream("../audio/drums.wav", fft_size=512, buffer_size=512)
        dot.no_stroke()

    def draw(self):
        dot.background(dot.black)
        # Sample 16 evenly-spaced FFT bins across the spectrum
        bins  = dot.music.fft()[:256:16]
        bar_w = dot.width // len(bins)
        for i, val in enumerate(bins):
            h = min(int(val * dot.height * 4), dot.height)
            dot.fill((int(min(val * 255 * 4, 255)), 100, 200))
            dot.rectangle((i * bar_w, dot.height - h), (i * bar_w + bar_w - 4, dot.height))

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
