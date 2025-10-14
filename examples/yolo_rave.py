

import os 
import cv2
import torch
import numpy as np

from math import floor
from ultralytics import YOLO

# Import util functions from the files in the 'src' directory
from dorothy import Dorothy
from dorothy.utils.yolo_draw_utils import draw_skeleton
from dorothy.utils.latent_util import create_latent_interp, clamp
from dorothy.utils.rave_download import download_pretrained_rave

# Load YOLO model
model = YOLO('model/yolov8n-pose.pt') 

# Turn off gradient tracking as we are not training any models
torch.set_grad_enabled(False)

# Class for sample latent space with RAVE
class RAVE_latent_generator:
    # Contructor that gets called when initialised
    # This creates a random latent interpolation
    def __init__(self, latent_dim, interp_len):
        self.latent_dim = latent_dim
        self.interp_len = interp_len
        self.latent_interp = create_latent_interp(intervals=self.interp_len, z_dim=self.latent_dim)
    
    # Sample a point on the existing latent interpolation
    # This function assumes the variable pos is in the range 0>=pos>=1
    def sample_latent(self, pos):
        print(pos)
        # Get position in latent interp array
        index_pos = pos * self.interp_len
        index_pos = clamp(0, index_pos, self.interp_len-1)
        index_pos = floor(index_pos)
        # Get latent from latent interpolation array
        latent = self.latent_interp.tolist()[index_pos]
        # Convert to tensor and reshape for RAVE input
        latent = torch.tensor(latent)
        # This changes the shape from (128) to (1,128,1)
        latent = latent.unsqueeze(0).unsqueeze(2)
        # Return the latent variable
        return latent

dot = Dorothy()

class MySketch:
    
    # Set up camera
    camera = cv2.VideoCapture(0)

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        # Turn off autofocus
        rave_model_dir = 'models/'

        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        #download_pretrained_rave(rave_model_dir)
        
        # Choose from:
        # - "vitange.ts", rave_latent_dim = 8
        # - "percussion.ts", rave_latent_dim = 8
        # - "VCTK.ts", rave_latent_dim = 8
        self.rave_latent_dim = 8
        # dot.music.start_rave_stream(os.path.join(rave_model_dir, "vintage.ts"), latent_dim=self.rave_latent_dim)
        self.rave_id = dot.music.start_rave_stream("models/rave_gospel.ts", latent_dim=self.rave_latent_dim)

        
        # Class for controlled sampling RAVE latent space
        self.latent_generator = RAVE_latent_generator(self.rave_latent_dim, 512)
        starting_latent = self.latent_generator.sample_latent(np.random.random())
        
        # Start of with the latent in the middle of our interpolation
        dot.music.audio_outputs[self.rave_id].current_latent = starting_latent
        dot.music.play()

    
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
                
                # Draw skeleton
                for pose in pose_list:
                    camera_feed = draw_skeleton(camera_feed, pose.squeeze())

                results_keypoint = results[0].keypoints.xyn.numpy()
                for result_keypoint in results_keypoint:
                    print(result_keypoint.shape)
                    if len(result_keypoint) == 17:
                        right_wrist = result_keypoint[10,:]
                        # Check that we have detected a right wrist
                        if np.all(right_wrist):
                            # Get the y position (height) of the right wrist
                            right_wrist_y = right_wrist[1]
                            # Sample our latent interpolation based on position of wrist
                            new_latent = self.latent_generator.sample_latent(right_wrist_y)
                            dot.music.audio_outputs[self.rave_id].current_latent = new_latent
   
            dot.canvas = camera_feed

             
MySketch()  
