from dorothy import Dorothy

dot = Dorothy(1080,480)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    def setup(self):
        file_path = "../audio/SO_SGC_90_get_to_heaven_alto_C.wav"
        file_path = "../audio/drums.wav"
        file_path = "../audio/metronome_90.wav"
        o = dot.music.start_file_stream(file_path, fft_size=512)
        # # 
        # o = dot.music.start_device_stream(1)
        # dot.music.audio_outputs[o].onset_detector.threshold = 0.5 
        # dot.music.audio_outputs[o].analyse_onsets = True
        self.show_beat = 0
        
    def draw(self):
        col = dot.black
        if dot.music.is_onset():
            self.show_beat = 2
        
        if self.show_beat > 0:
            col = dot.green
        
        dot.background(col)
        self.show_beat -= 1
        
        dot.draw_waveform(0, with_playhead=True)
        pixels_per_sample = dot.width/len(dot.music.audio_outputs[0].y[0])
        # print(pixels_per_sample)
        dot.stroke(dot.red)
        dot.set_stroke_weight(1)
        for onset in dot.music.audio_outputs[0].onsets:
            dot.line((onset*pixels_per_sample, 0), (onset*pixels_per_sample, dot.height))

MySketch()   
    







