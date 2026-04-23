"""Read overall amplitude (RMS) from a playing audio stream."""
from dorothy import Dorothy

dot = Dorothy(640, 480)

class MySketch:

    def setup(self):
        dot.music.start_file_stream("../audio/disco.wav", fft_size=512)

    def draw(self):
        dot.background(dot.white)
        dot.fill(dot.red)
        # amplitude() returns 0.0–1.0 RMS of the current buffer
        dot.circle((dot.width // 2, dot.height // 2),
                   int(dot.music.amplitude() * dot.height))

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
