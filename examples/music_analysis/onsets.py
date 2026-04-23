"""Detect note onsets and mark them on the waveform display."""
from dorothy import Dorothy

dot = Dorothy(1080, 480)

class MySketch:

    def setup(self):
        o = dot.music.start_file_stream("../audio/drums.wav", fft_size=512)
        # Enable online onset detection on this stream
        dot.music.audio_outputs[o].analyse_onsets = True
        self.flash = 0

    def draw(self):
        if dot.music.is_onset():
            self.flash = 3

        dot.background(dot.black if self.flash == 0 else dot.green)
        self.flash = max(0, self.flash - 1)

        dot.draw_waveform(0, with_playhead=True)

        # Draw a vertical line at each detected onset sample position
        stream = dot.music.audio_outputs[0]
        if len(stream.y[0]) > 0:
            px_per_sample = dot.width / len(stream.y[0])
            dot.stroke(dot.red)
            dot.set_stroke_weight(1)
            for onset in stream.onsets:
                x = onset * px_per_sample
                dot.line((x, 0), (x, dot.height))

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
