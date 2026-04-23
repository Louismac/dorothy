"""Live-coding with audio: replace the synth callback in draw() each frame.

Changes to mouse position take effect sample-accurately without a file save,
because the callback is swapped every frame rather than only on hot-reload.
"""
import numpy as np
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def setup(self):
        self.phase = 0.0
        self.sr    = 44100

        def get_frame(size):
            return np.zeros(size)

        index = dot.music.start_dsp_stream(get_frame, sr=self.sr)
        self.chroma = np.zeros(12)

        def on_new_frame(buffer=np.zeros(2048)):
            import librosa
            c = librosa.feature.chroma_stft(y=buffer, n_fft=512)
            self.chroma = c.mean(axis=1)

        dot.music.audio_outputs[index].on_new_frame = on_new_frame
        dot.stroke(dot.red)
        dot.set_stroke_weight(10)

    def draw(self):
        # Swap the callback every frame — mouse parameters update in real time
        def get_frame(size):
            freq  = dot.mouse_x
            amp   = dot.mouse_y / dot.height
            delta = 2 * np.pi * freq / self.sr
            audio = amp * np.sin(self.phase + delta * np.arange(size))
            self.phase += delta * size
            return audio

        dot.music.audio_outputs[0].get_frame = get_frame

        dot.background(dot.white)
        for i, c in enumerate(self.chroma):
            x = (dot.width // 12) * i
            y = int(dot.height - dot.height * c)
            dot.line((x, dot.height), (x, y))

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
