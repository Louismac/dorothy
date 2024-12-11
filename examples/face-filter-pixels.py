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
        self.face_cascade=cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
    
    def draw(self):
        #Clear background
        dot.background(dot.black)
        success, camera_feed = self.camera.read()
        
        if success:
            camera_feed = cv2.resize(camera_feed,(dot.width, dot.height))
            camera_feed = cv2.cvtColor(camera_feed, cv2.COLOR_BGR2RGB)
            camera_feed_grayscale = cv2.cvtColor(camera_feed, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(camera_feed_grayscale, 1.1, 4)
            
            #Loop over faces
            for face_x, face_y, face_w, face_h in faces:
                #Chop out the face
                face_pixels = camera_feed[face_y:face_y+face_h,face_x:face_x+face_w].copy()
                skip = 12
                #Downsample the face
                downsampled = face_pixels[::skip,::skip]
                radius = 6
                for i in range(downsampled.shape[0]):
                    for j in range(downsampled.shape[1]):
                        #Draw circles in place of the downsampled pixels
                        colour = (int(downsampled[j][i][0]),int(downsampled[j][i][1]),int(downsampled[j][i][2]))
                        x = face_x+(i*skip)
                        y = face_y+(j*skip)
                        dot.fill(colour)
                        dot.circle((x,y),radius)
                    
MySketch()          







