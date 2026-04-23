"""Detect beats and flash the background on each one."""
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def setup(self):
        o = dot.music.start_file_stream("../audio/disco.wav", fft_size=512)
        # Enable beat tracking on this stream
        dot.music.audio_outputs[o].analyse_beats = True
        self.flash = 0

    def draw(self):
        if dot.music.is_beat():
            self.flash = 10          # hold the flash for 10 frames
        dot.background(dot.white if self.flash > 0 else dot.black)
        self.flash = max(0, self.flash - 1)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
