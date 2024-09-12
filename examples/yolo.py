import cv2
import torch

from ultralytics import YOLO

from dorothy import Dorothy
from dorothy.utils.yolo_draw_utils import draw_skeleton

model = YOLO('models/yolov8n-pose.pt') 
dot = Dorothy()


class MySketch:
    
    # Set up camera
    camera = cv2.VideoCapture(0)

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        # Turn off autofocus
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    
    def draw(self):
        # Pull in frame
        success, camera_feed = self.camera.read()
        if success:
            # Convert color
            camera_feed = cv2.cvtColor(camera_feed, cv2.COLOR_BGR2RGB)
            # Resize to canvas size
            camera_feed = cv2.resize(camera_feed,(dot.width, dot.height))
            
            # Process frame with YOLO model
            results = model(camera_feed)
            
            if results[0].keypoints != None:
                # Get skeleton keypoints from YOLO results
                poses = results[0].keypoints.data
                pose_list = torch.split(poses,1,0)
                
                for pose in pose_list:
                    camera_feed = draw_skeleton(camera_feed, pose.squeeze())
   
            dot.canvas = camera_feed
             
MySketch()          







