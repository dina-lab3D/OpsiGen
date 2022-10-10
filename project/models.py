## Standard libraries

## Imports for plotting

import matplotlib
import torch

matplotlib.rcParams['lines.linewidth'] = 2.0
## PyTorch

import torch.nn as nn
import torch.nn.functional as F

# Torchvision
# import torch_geometric.nn as geom_nn
# from torch_geometric.nn.norm import BatchNorm


class CryoNet(nn.Module):

    def __init__(self, threas):
        super().__init__()
        self.kernel_size = 3
        layer_size = int((threas * 2 - (self.kernel_size - 1)) / 2)
        self.conv_layer1 = self._conv_layer_set(1, 1)
        self.fc1 = nn.Linear(layer_size, 50)
        self.conv_layer2 = self._conv_layer_set(1, 1)
        beginning_layer_size = int((layer_size - (self.kernel_size - 1)) / 2)
        last_layer_size = int((50 - (self.kernel_size - 1)) / 2)
        self.fc2 = nn.Linear(beginning_layer_size * beginning_layer_size * last_layer_size, 200)
        self.activation1 = nn.LeakyReLU()
        self.fc3 = nn.Linear(200, 100)

    def forward(self, x):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 0)
        out = self.conv_layer1(x)
        out = self.fc1(out)
        out = self.conv_layer2(out)
        out = self.fc2(out.flatten(start_dim=1))
        out = self.activation1(out)
        out = self.fc3(out)
        return F.normalize(out, dim=1)

    def _conv_layer_set(self, in_c, out_c):
        """
        Decrease 5 from any axis
        :param in_c: in channels
        :param out_c: out channels
        :return:
        """
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size)),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

