import torch
import torch.nn as nn
import numpy as np

def slerp(val, low, high):
    '''
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    '''
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

def clamp(minimum, x, maximum):
    return max(minimum, min(x, maximum))

def create_latent_interp(intervals, z_dim):
    num_interp = intervals
    interp_vals = np.linspace(0, 1, num=num_interp)
    latent_a = torch.randn(z_dim)
    latent_b = torch.randn(z_dim)
    latent_a_np = latent_a.cpu().numpy().squeeze()
    latent_b_np = latent_b.cpu().numpy().squeeze()
    latent_interp = np.array([slerp(v, latent_a_np, latent_b_np) for v in interp_vals],
                                    dtype=np.float32)
    return latent_interp