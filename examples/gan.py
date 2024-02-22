import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

from src.Dorothy import Dorothy
from utils.util import clamp, create_latent_interp
from utils.gan_model import Generator

# GAN generator hyperparams
image_size = 32  
num_channels = 3 
z_dim = 100 
n_f_maps = 32
load_path = 'models/gan_weights_mnist_1000_epochs.pt'

# Width and height of canvas
WIDTH = 128
HEIGHT = 128

dot = Dorothy(WIDTH,HEIGHT)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        # Load generator network
        self.generator = Generator(z_dim,n_f_maps,num_channels)
        checkpoint_sd = torch.load(load_path, map_location=torch.device('cpu'))
        self.generator.load_state_dict(checkpoint_sd['generator'])
        self.generator.eval()
        # Create latent interpolation array
        self.latent_interp = create_latent_interp(intervals=WIDTH, z_dim=z_dim)
        
    def draw(self):
        # Get mouse X position
        latent_index = dot.mouse_x
        # Make sure it is in valid range
        latent_index = clamp(0, latent_index, WIDTH)
        # Get latent vector from array of latent interpolations
        latent = self.latent_interp.tolist()[latent_index]
        # Convert to torch tensor and reshape
        latent_tensor = torch.tensor(latent)
        latent_tensor = latent_tensor.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # Create image 
        image_tensor = self.generator(latent_tensor)
        # Convert and upscale image
        image = transforms.functional.to_pil_image(image_tensor.squeeze(0)).convert('RGB')
        image_upscaled = cv2.resize(np.asarray(image), (128,128), interpolation= cv2.INTER_LINEAR)
        # Set background to black
        dot.background((0,0,0))
        # Copy image into canvas
        dot.paste(dot.canvas, np.asarray(image_upscaled))

MySketch()  