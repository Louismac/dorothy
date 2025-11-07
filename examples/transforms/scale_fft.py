from dorothy import Dorothy
from cv2 import line
import numpy as np
from PIL import Image

dot = Dorothy(640, 480)

class MySketch:
  
  def __init__(self):
    dot.start_loop(self.setup, self.draw)
  
  def setup(self):
    file_path = "../audio/drums.wav"
    self.rgb_images = np.array([np.array(Image.open(f'../images/aligned_faces/face_{i}.jpg')) for i in range(8)])
    dot.music.start_file_stream(file_path, fft_size=512, buffer_size=512)
  
  def draw(self):

    dot.background(dot.black)
    dot.fill(dot.red)
    dot.stroke(dot.red)
    for bin_num, bin_val in enumerate(dot.music.fft()[:256:8]):
      x = bin_num*50
      with dot.transform():
        dot.scale(1,bin_val)
        dot.paste(self.rgb_images[bin_num%8], (x, 0))
        


MySketch() 