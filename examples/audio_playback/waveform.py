"""Draw the audio waveform with a playhead; map mouse X to gain."""
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def setup(self):
        dot.music.start_file_stream("../audio/bass.wav")

    def draw(self):
        dot.background(dot.white)
        dot.draw_waveform(col=dot.black, with_playhead=True)
        # Map mouse position across the full window width to gain 0–1
        dot.music.audio_outputs[0].gain = dot.mouse_x / dot.width

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
