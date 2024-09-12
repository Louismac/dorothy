import torch
import torch.nn as nn
import torch.nn.functional as F
## Try writing comments where a hash has been left
## What are the main differences between the generator and discriminator class?
## Search the pytorch reference for classes you have not seen before:
## https://pytorch.org/docs/stable/index.html

#
class Generator(nn.Module):
    #
    def __init__(self, z_dim, n_f_maps, num_channels):
        super(Generator, self).__init__()
        
        #
        self.conv1 = nn.ConvTranspose2d(z_dim, n_f_maps * 4, kernel_size=4, stride=1, padding=0, bias=False)
        self.conv2 = nn.ConvTranspose2d( n_f_maps * 4, n_f_maps * 2, 4, 2, 1, bias=False)
        self.conv3 = nn.ConvTranspose2d( n_f_maps * 2, n_f_maps, 4, 2, 1, bias=False)
        self.conv4 = nn.ConvTranspose2d( n_f_maps, num_channels, 4, 2, 1, bias=False)

        #
        self.bn1 = nn.BatchNorm2d(n_f_maps * 4)
        self.bn2 = nn.BatchNorm2d(n_f_maps * 2)
        self.bn3 = nn.BatchNorm2d(n_f_maps)

    #
    def forward(self, x):
        #
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.tanh(self.conv4(x))
        return x

#  
class Discriminator(nn.Module):
    #
    def __init__(self, n_f_maps, num_channels):
        super(Discriminator, self).__init__()
        
        #
        self.conv1 = nn.Conv2d(num_channels, n_f_maps, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(n_f_maps, n_f_maps * 2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(n_f_maps * 2, n_f_maps * 4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(n_f_maps * 4, 1, 4, 1, 0, bias=False)

        #
        self.bn1 = nn.BatchNorm2d(n_f_maps * 2)
        self.bn2 = nn.BatchNorm2d(n_f_maps * 4)

    #
    def forward(self, x):
        #
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.bn1(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.bn2(x)
        x = F.sigmoid(self.conv4(x))
        return x
    
