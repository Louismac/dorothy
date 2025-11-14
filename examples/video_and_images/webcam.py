import cv2
from cv2 import circle
from dorothy import Dorothy

dot = Dorothy()

class MySketch:
    
    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        self.camera = cv2.VideoCapture(0)
    
    def draw(self):
        #Clear background
        dot.background(dot.black)
        success, camera_feed = self.camera.read()
        
        if success:
            camera_feed = cv2.resize(camera_feed,(dot.width, dot.height))
            camera_feed = cv2.cvtColor(camera_feed, cv2.COLOR_BGR2RGB)
            dot.paste(camera_feed,(0,0))
                    
MySketch()          







