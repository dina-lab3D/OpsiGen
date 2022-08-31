## Standard libraries

## Imports for plotting

import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
## PyTorch

import torch.nn as nn
import torch.nn.functional as F

# Torchvision
import torch_geometric.nn as geom_nn
from torch_geometric.nn.norm import BatchNorm


class CryoNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_layer1 = self._conv_layer_set(1, 3)
        self.fc1 = nn.Linear(4, 50)
        self.conv_layer2 = self._conv_layer_set(3, 4)
        self.fc2 = nn.Linear(96, 30)

    def forward(self, x):
        if x.shape != (1, 10, 10, 10):
            breakpoint()
        out = self.conv_layer1(x)
        out = self.fc1(out)
        out = self.conv_layer2(out)
        out = self.fc2(out.flatten())
        return out.flatten()

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3)),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer


class ProteinNet(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.fc1 = nn.Linear(c_in, c_out)
        # self.activation1 = nn.LeakyReLU()
        # self.fc2 = nn.Linear(45, c_out)

    def forward(self, x):
        out = self.fc1(x)
        # out = self.activation1(out)
        # out = self.fc2(out)
        return out.flatten()
