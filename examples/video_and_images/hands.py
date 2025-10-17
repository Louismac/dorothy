#developed from https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
import cv2
import mediapipe as mp
from dorothy import Dorothy

dot = Dorothy()

class MySketch:
    
    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        self.camera = cv2.VideoCapture(0)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def draw(self):

        success, camera_feed = self.camera.read()
        if success:
            
            camera_feed = cv2.resize(camera_feed,(dot.width, dot.height))
            camera_feed = cv2.cvtColor(camera_feed, cv2.COLOR_BGR2RGB)
            results = self.hands.process(camera_feed)

            colours = [dot.red, dot.green, dot.blue, dot.yellow, dot.purple]
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    lm = hand_landmarks.landmark
                    #Five fingers
                    for i in range(5):
                        dot.set_stroke_weight(3)
                        dot.stroke(colours[i])
                        index = 1 + (i*4)
                        #Four points on each finger
                        for j in range(3):
                            pts = ((lm[index+j].x*dot.width,lm[index+j].y*dot.height),
                                  (lm[index+j+1].x*dot.width,lm[index+j+1].y*dot.height))
                            dot.line(pts[0],pts[1],camera_feed)

            camera_feed = cv2.flip(camera_feed, 1)
            dot.paste(camera_feed, (0,0))
        
MySketch()          
