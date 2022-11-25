import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import numpy as np

# Torchvision
import torch_geometric.nn as geom_nn
from torch_geometric.nn.norm import BatchNorm


class SimpleModel(nn.Module):

    def __init__(self):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of hidden layers
            dp_rate - Dropout rate to apply throughout the network
        """
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(26, 24), nn.ReLU(inplace=True), nn.Dropout(0.1))
        self.l2 = nn.Sequential(nn.Linear(24, 20), nn.ReLU(inplace=True), nn.Dropout(0.1))
        self.l3 = nn.Sequential(nn.Linear(20, 10), nn.ReLU(inplace=True), nn.Dropout(0.1))
        self.l4 = nn.Linear(240, 1)

    def forward(self, x, *args, **kwargs):
        """
        Inputs:
            x - Input features per node
        """
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(torch.flatten(x3, start_dim=1))
        return x4.squeeze()

