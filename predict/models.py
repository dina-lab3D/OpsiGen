import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import numpy as np

# Torchvision
import torch_geometric.nn as geom_nn
from torch_geometric.nn.norm import BatchNorm


class MLPModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, num_layers=2, dp_rate=0.1):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of hidden layers
            dp_rate - Dropout rate to apply throughout the network
        """
        super().__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [nn.Linear(in_channels, c_out)]
        self.layers = nn.Sequential(*layers)
        self.device = device

    def forward(self, x, *args, **kwargs):
        """
        Inputs:
            x - Input features per node
        """
        return self.layers(x)


class GATModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        self.num_convs = 5
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv2 = GAT(c_hidden, c_hidden, edge_dim=edge_dim)
        self.conv3 = GAT(c_hidden, c_out, edge_dim=edge_dim, concat=False)
        self.l1 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(c_out, 10)
        )
        self.l2 = nn.Sequential(
            nn.Linear(10, 1)
        )
        # self.l3 = nn.Sequential(
        #     nn.Dropout(dp_rate),
        #     nn.Linear(c_out * 2, c_out * 1)
        # )
        # self.l4 = nn.Linear(c_out, 1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 36
        # edge_weights[:, -20:,-20:] = 0
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x1)
        x2 = self.conv2(x1, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x2)
        x3 = self.conv3(x2, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x3)
        x4, _ = torch.max(x3, dim=-2)
        x5 = self.l1(x4)
        nn.ReLU(inplace=True)(x5)
        x6 = self.l2(x5)
        # nn.ReLU(inplace=True)(x6)
        # x7 = self.l3(x6)
        # nn.ReLU(inplace=True)(x7)
        # x8 = self.l4(x7)
        # x5 = torch.mean(input=x2, axis=-2)
        # x5 = torch.max(self.head(x3))

        return x6


class GAT2Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        self.num_convs = 5
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv2 = GAT(c_hidden, c_hidden, edge_dim=edge_dim)
        self.conv3 = GAT(c_hidden, c_out, edge_dim=edge_dim, concat=False)
        self.l1 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(c_out, 10)
        )
        self.l2 = nn.Sequential(
            nn.Linear(10, 1)
        )
        # self.l3 = nn.Sequential(
        #     nn.Dropout(dp_rate),
        #     nn.Linear(c_out * 2, c_out * 1)
        # )
        # self.l4 = nn.Linear(c_out, 1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 36
        # edge_weights[:, -20:,-20:] = 0
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x1)
        x2 = self.conv2(x1, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x2)
        x3 = self.conv3(x2, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x3)
        x4 = torch.mean(x3, dim=-2)
        x5 = self.l1(x4)
        nn.ReLU(inplace=True)(x5)
        x6 = self.l2(x5)
        # nn.ReLU(inplace=True)(x6)
        # x7 = self.l3(x6)
        # nn.ReLU(inplace=True)(x7)
        # x8 = self.l4(x7)
        # x5 = torch.mean(input=x2, axis=-2)
        # x5 = torch.max(self.head(x3))

        return x6

class GAT3Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        self.num_convs = 5
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv2 = GAT(c_hidden, c_hidden, edge_dim=edge_dim)
        self.conv3 = GAT(c_hidden, c_out, edge_dim=edge_dim, concat=False)
        self.l1 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(c_out, int(c_out / 2))
        )
        self.l2 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(int(c_out / 2), int(c_out / 4))
        )
        self.l3 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(int(c_out / 4), int(c_out / 8))
        )
        self.l4 = nn.Linear(int(c_out / 8), 1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 57
        # edge_weights[:, -20:,-20:] = 0
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x1)
        x2 = self.conv2(x1, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x2)
        x3 = self.conv3(x2, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x3)
        x4 = torch.mean(x3, dim=-2)
        x5 = self.l1(x4)
        nn.ReLU(inplace=True)(x5)
        x6 = self.l2(x5)
        nn.ReLU(inplace=True)(x6)
        x7 = self.l3(x6)
        nn.ReLU(inplace=True)(x7)
        x8 = self.l4(x7)
        # x5 = torch.mean(input=x2, axis=-2)
        # x5 = torch.max(self.head(x3))

        return x8

class GAT4Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        self.num_convs = 5
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv3 = GAT(c_hidden, c_out, edge_dim=edge_dim, concat=False)
        self.l1 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(c_out, 10)
        )
        self.l2 = nn.Sequential(
            nn.Linear(10, 1)
        )
        # self.l3 = nn.Sequential(
        #     nn.Dropout(dp_rate),
        #     nn.Linear(c_out * 2, c_out * 1)
        # )
        # self.l4 = nn.Linear(c_out, 1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 37
        # edge_weights[:, -20:,-20:] = 0
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x1)
        x3 = self.conv3(x1, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x3)
        x4 = torch.mean(x3, dim=-2)
        x5 = self.l1(x4)
        nn.ReLU(inplace=True)(x5)
        x6 = self.l2(x5)
        # nn.ReLU(inplace=True)(x6)
        # x7 = self.l3(x6)
        # nn.ReLU(inplace=True)(x7)
        # x8 = self.l4(x7)
        # x5 = torch.mean(input=x2, axis=-2)
        # x5 = torch.max(self.head(x3))

        return x6

class GAT5Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv3 = GAT(c_hidden, c_out, edge_dim=edge_dim, concat=False)
        self.l1 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(c_out, int(c_out / 2))
        )
        self.l2 = nn.Sequential(
            nn.Linear(int(c_out / 2), 1)
        )
        # self.l3 = nn.Sequential(
        #     nn.Dropout(dp_rate),
        #     nn.Linear(c_out * 2, c_out * 1)
        # )
        # self.l4 = nn.Linear(c_out, 1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 36
        # edge_weights[:, -20:,-20:] = 0
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x1)
        x3 = self.conv3(x1, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x3)
        x4 = torch.mean(x3, dim=-2)
        x5 = self.l1(x4)
        nn.ReLU(inplace=True)(x5)
        x6 = self.l2(x5)

        return x6


class GAT6Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv2 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv3 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv4 = GAT(c_hidden, c_out, edge_dim=edge_dim, concat=False)
        self.l1 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(c_out, int(c_out / 2))
        )
        self.l2 = nn.Sequential(
            nn.Linear(int(c_out / 2), 1)
        )

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 36
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        x1 = self.normalize(x1)
        x2 = self.conv2(x1, edge_index, flatten_weights)
        x2 = self.normalize(x2)
        x3 = self.conv3(x2, edge_index, flatten_weights)
        x3 = self.normalize(x3)
        x4 = self.conv4(x3, edge_index, flatten_weights)
        x5 = torch.mean(x4, dim=-2)
        x6 = self.l1(x5)
        nn.ReLU(inplace=True)(x6)
        x7 = self.l2(x6)

        return x7

class GAT7Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv2 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv3 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv4 = GAT(c_hidden, c_out, edge_dim=edge_dim, concat=False)
        self.l1 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(c_out, int(c_out / 2))
        )
        self.l2 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(int(c_out / 2), int(c_out / 4))
        )
        self.l3 = nn.Sequential(
            nn.Linear(int(c_out / 4), 1)
        )
        # self.l3 = nn.Sequential(
        #     nn.Dropout(dp_rate),
        #     nn.Linear(c_out * 2, c_out * 1)
        # )
        # self.l4 = nn.Linear(c_out, 1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 36
        # edge_weights[:, -20:,-20:] = 0
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x1)
        x2 = self.conv2(x1, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x2)
        x3 = self.conv3(x2, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x3)
        x4 = self.conv4(x3, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x4)
        x5 = torch.mean(x4, dim=-2)
        x6 = self.l1(x5)
        nn.ReLU(inplace=True)(x6)
        x7 = self.l2(x6)
        nn.ReLU(inplace=True)(x7)
        x8 = self.l3(x7)

        return x8

class GAT8Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv2 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv3 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv4 = GAT(c_hidden, c_out, edge_dim=edge_dim, concat=False)
        self.c1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, dilation=3)
        conv_size1 = int((c_out - 3 * 4 - 1)) + 1
        self.c2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, dilation=3)
        conv_size2 = int((conv_size1 - 3 * (5-1) - 1)) + 1
        self.c3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, dilation=3, stride=2)
        conv_size3 = int((conv_size2 - 3 * (5-1) - 1) / 2) + 1
        self.l1 = nn.Linear(conv_size3, 1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 36
        # edge_weights[:, -20:,-20:] = 0
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x1)
        x2 = self.conv2(x1, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x2)
        x3 = self.conv3(x2, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x3)
        x4 = self.conv4(x3, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x4)
        x5 = torch.mean(x4, dim=-2).unsqueeze(dim=0)
        x6 = self.c1(x5)
        nn.ReLU(inplace=True)(x6)
        x7 = self.c2(x6)
        nn.ReLU(inplace=True)(x7)
        x8 = self.c3(x7)
        nn.ReLU(inplace=True)(x8)
        x9 = self.l1(x8)


        return x9

class GAT9Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv2 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv3 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv4 = GAT(c_hidden, c_out, edge_dim=edge_dim, concat=False)
        self.c1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, dilation=3)
        conv_size1 = int((c_out - 3 * 4 - 1)) + 1
        self.c2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, dilation=3)
        conv_size2 = int((conv_size1 - 3 * (5-1) - 1)) + 1
        self.c3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, dilation=3, stride=2)
        conv_size3 = int((conv_size2 - 3 * (5-1) - 1) / 2) + 1
        self.l1 = nn.Linear(conv_size3, 1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 36
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        x1 = ((x1 - torch.mean(x1, dim=-2)) / torch.std(x1, dim=-2)) + 1
        x2 = self.conv2(x1, edge_index, flatten_weights)
        x2 = ((x2 - torch.mean(x2, dim=-2)) / torch.std(x2, dim=-2)) + 1
        x3 = self.conv3(x2, edge_index, flatten_weights)
        x3 = ((x3 - torch.mean(x3, dim=-2)) / torch.std(x3, dim=-2)) + 1
        x4 = self.conv4(x3, edge_index, flatten_weights)

        x5 = torch.mean(x4, dim=-2).unsqueeze(dim=0)
        x6 = self.c1(x5)
        nn.ReLU(inplace=True)(x6)
        x7 = self.c2(x6)
        nn.ReLU(inplace=True)(x7)
        x8 = self.c3(x7)
        nn.ReLU(inplace=True)(x8)
        x9 = self.l1(x8)


        return x9

class GAT9Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv2 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv3 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv4 = GAT(c_hidden, c_out, edge_dim=edge_dim, concat=False)
        self.c1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, dilation=3)
        conv_size1 = int((c_out - 3 * (10-1) - 1)) + 1
        self.c2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, dilation=3)
        conv_size2 = int((conv_size1 - 3 * (10-1) - 1)) + 1
        self.c3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, dilation=3, stride=2)
        conv_size3 = int((conv_size2 - 3 * (10-1) - 1) / 2) + 1
        self.l1 = nn.Linear(conv_size3, 1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 36
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        x1 = ((x1 - torch.mean(x1, dim=-2)) / torch.std(x1, dim=-2)) + 1
        x2 = self.conv2(x1, edge_index, flatten_weights)
        x2 = ((x2 - torch.mean(x2, dim=-2)) / torch.std(x2, dim=-2)) + 1
        x3 = self.conv3(x2, edge_index, flatten_weights)
        x3 = ((x3 - torch.mean(x3, dim=-2)) / torch.std(x3, dim=-2)) + 1
        x4 = self.conv4(x3, edge_index, flatten_weights)

        x5 = torch.mean(x4, dim=-2).unsqueeze(dim=0)
        x6 = self.c1(x5)
        nn.ReLU(inplace=True)(x6)
        x7 = self.c2(x6)
        nn.ReLU(inplace=True)(x7)
        x8 = self.c3(x7)
        nn.ReLU(inplace=True)(x8)
        x9 = self.l1(x8)


        return x9

class GAT10Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv2 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv3 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv4 = GAT(c_hidden, c_out, edge_dim=edge_dim, concat=False)
        self.l1 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(c_out, int(c_out / 2))
        )
        self.l2 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(int(c_out / 2), int(c_out / 4))
        )
        self.l3 = nn.Sequential(
            nn.Linear(int(c_out / 4), 1)
        )

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 36
        # edge_weights[:, -20:,-20:] = 0
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x1)
        x2 = self.conv2(x1, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x2)
        x3 = self.conv3(x2, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x3)
        x4 = self.conv4(x3, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x4)
        x5 = torch.mean(x4, dim=-2)
        x6 = self.l1(x5)
        nn.ReLU(inplace=True)(x6)
        x7 = self.l2(x6)
        nn.ReLU(inplace=True)(x7)
        x8 = self.l3(x7)

        return x8

class GAT11Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv2 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv3 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv4 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv5 = GAT(c_hidden, c_out, edge_dim=edge_dim, concat=False)
        self.l1 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(c_out, int(c_out / 2))
        )
        self.l2 = nn.Sequential(
            nn.Linear(int(c_out / 2), 1)
        )

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 36
        # edge_weights[:, -20:,-20:] = 0
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x1)
        x2 = self.conv2(x1, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x2)
        x3 = self.conv3(x2, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x3)
        x4 = self.conv4(x3, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x4)
        x5 = self.conv4(x4, edge_index, flatten_weights)
        nn.ReLU(inplace=True)(x5)
        x6 = torch.mean(x5, dim=-2)
        x7 = self.l1(x6)
        nn.ReLU(inplace=True)(x7)
        x8 = self.l2(x7)

        return x8

class GAT12Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_out, edge_dim=edge_dim)
        self.l1 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(c_out, int(c_out / 2))
        )
        self.l2 = nn.Sequential(
            nn.Linear(int(c_out / 2), 1)
        )

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 36
        # edge_weights[:, -20:,-20:] = 0
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        x2 = torch.mean(x1, dim=-2)
        x3 = self.l1(x2)
        nn.ReLU(inplace=True)(x3)
        x3 = self.l2(x3)

        return x3

class GAT13Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv2 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv3 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv4 = GAT(c_hidden, 1, edge_dim=edge_dim, concat=False)

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 36
        # edge_weights[:, -20:,-20:] = 0
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        x1 = self.normalize(x1)
        x2 = self.conv2(x1, edge_index, flatten_weights)
        x2 = self.normalize(x2)
        x3 = self.conv3(x2, edge_index, flatten_weights)
        x3 = self.normalize(x3)
        x4 = self.conv4(x3, edge_index, flatten_weights)
        x5 = torch.sum(x4, dim=-2)

        return x5

class GAT14Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv2 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv3 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv4 = GAT(c_hidden, 1, edge_dim=edge_dim, concat=False)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 36
        # edge_weights[:, -20:,-20:] = 0
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        x2 = self.conv2(x1, edge_index, flatten_weights)
        x3 = self.conv3(x2, edge_index, flatten_weights)
        x4 = self.conv4(x3, edge_index, flatten_weights)
        x5 = torch.sum(x4, dim=-2)

        return x5

class GAT15Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv2 = GAT(c_hidden, 1, edge_dim=edge_dim, concat=False)

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 36
        # edge_weights[:, -20:,-20:] = 0
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        x1 = self.normalize(x1)
        x2 = self.conv2(x1, edge_index, flatten_weights)
        x3 = torch.mean(x2, dim=-2)

        return x3

class GAT16Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv2 = GAT(c_hidden, c_hidden, edge_dim=edge_dim)
        self.conv3 = GAT(c_hidden, c_hidden, edge_dim=edge_dim)
        self.conv4 = GAT(c_hidden, c_hidden, edge_dim=edge_dim)
        self.conv5 = GAT(c_hidden, c_hidden, edge_dim=edge_dim)
        self.conv6 = GAT(c_hidden, c_hidden, edge_dim=edge_dim)
        self.conv7 = GAT(c_hidden, c_hidden, edge_dim=edge_dim)
        self.conv8 = GAT(c_hidden, 1, edge_dim=edge_dim, concat=False)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        x2 = self.conv2(x1, edge_index, flatten_weights)
        x3 = self.conv3(x2, edge_index, flatten_weights)
        x4 = self.conv4(x3, edge_index, flatten_weights)
        x5 = self.conv5(x4, edge_index, flatten_weights)
        x6 = self.conv6(x5, edge_index, flatten_weights)
        x7 = self.conv7(x6, edge_index, flatten_weights)
        x8 = self.conv8(x7, edge_index, flatten_weights)
        x9 = torch.mean(x8, dim=-2)

        return x9

class GAT17Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv2 = GAT(c_hidden, c_hidden, edge_dim=edge_dim)
        self.conv3 = GAT(c_hidden, 1, edge_dim=edge_dim, concat=False)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 36
        # edge_weights[:, -20:,-20:] = 0
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        x1 = x1 / (torch.sum(x1, -1).unsqueeze(-1))
        x2 = self.conv2(x1, edge_index, flatten_weights)
        x2 = x2 / (torch.sum(x2, -1).unsqueeze(-1))
        x3 = self.conv3(x2, edge_index, flatten_weights)
        x4 = torch.mean(x3, dim=-2)

        return x4

class GAT18Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv2 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv3 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv4 = GAT(c_hidden, 1, edge_dim=edge_dim, concat=False)

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 36
        # edge_weights[:, -20:,-20:] = 0
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        x1 = self.normalize(x1)
        x2 = self.conv2(x1, edge_index, flatten_weights)
        x2 = self.normalize(x2)
        x3 = self.conv3(x2, edge_index, flatten_weights)
        x3 = self.normalize(x3)
        x4 = self.conv4(x3, edge_index, flatten_weights)
        x5 = torch.mean(x4, dim=-2)

        return x5

class GAT19Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_out, edge_dim=edge_dim)
        self.b1 = BatchNorm(c_out)
        self.conv2 = GAT(c_out, c_hidden, edge_dim=edge_dim, concat=False)
        self.b2 = BatchNorm(c_hidden)
        self.conv3 = GAT(c_hidden, c_out, edge_dim=edge_dim, concat=False)
        self.b3 = BatchNorm(c_out)
        self.conv4 = GAT(c_out, 1, edge_dim=edge_dim, concat=False)

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.b1(self.conv1(features, edge_index, flatten_weights))
        x2 = self.b2(self.conv2(x1, edge_index, flatten_weights))
        x3 = self.b3(self.conv3(x2, edge_index, flatten_weights))
        x4 = self.conv4(x3, edge_index, flatten_weights)
        x5 = torch.mean(x4, dim=-2)

        return x5

class GAT20Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = geom_nn.GCNConv(c_in, c_out, edge_dim=edge_dim, add_self_loops=False)
        self.b1 = BatchNorm(c_out)
        self.conv2 = geom_nn.GATv2Conv(c_out, c_hidden, edge_dim=edge_dim, add_self_loops=False)
        self.b2 = BatchNorm(c_hidden)
        self.conv3 = geom_nn.GATv2Conv(c_hidden, c_out, edge_dim=edge_dim, add_self_loops=False)
        self.b3 = BatchNorm(c_out)
        self.conv4 = geom_nn.GCNConv(c_out, 1, edge_dim=edge_dim, add_self_loops=False)

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.b1(self.conv1(features, edge_index, flatten_weights))
        x2 = self.b2(self.conv2(x1, edge_index, flatten_weights))
        x3 = self.b3(self.conv3(x2, edge_index, flatten_weights))
        x4 = self.conv4(x3, edge_index, flatten_weights)
        x5 = torch.mean(x4, dim=-2)

        return x5

class GAT21Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        self.device = device
        self.conv1 = geom_nn.GCNConv(c_in, c_out, edge_dim=edge_dim, add_self_loops=False)
        self.b1 = BatchNorm(c_out)
        self.conv2 = geom_nn.GATv2Conv(c_out, c_hidden, edge_dim=edge_dim, add_self_loops=False)
        self.b2 = BatchNorm(c_hidden)
        self.conv3 = geom_nn.GATv2Conv(c_hidden, c_out, edge_dim=edge_dim, add_self_loops=False)
        self.b3 = BatchNorm(c_out)
        self.conv4 = geom_nn.GCNConv(c_out, 1, edge_dim=edge_dim, add_self_loops=False)

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.b1(self.conv1(features, edge_index, flatten_weights))
        nn.ELU(x1, inplace=True)
        x2 = self.b2(self.conv2(x1, edge_index, flatten_weights))
        nn.ELU(x2, inplace=True)
        x3 = self.b3(self.conv3(x2, edge_index, flatten_weights))
        nn.ELU(x3, inplace=True)
        x4 = self.conv4(x3, edge_index, flatten_weights)
        x5 = torch.mean(x4, dim=-2)

        return x5


class GAT22Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.conv2 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv3 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.conv4 = GAT(c_hidden, c_out, edge_dim=edge_dim, concat=False)
        self.l1 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(c_out, int(c_out / 2))
        )
        self.l2 = nn.Sequential(
            nn.Linear(int(c_out / 2), 1)
        )

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        NUM_FEATURES = 36
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.conv1(features, edge_index, flatten_weights)
        x1 = self.normalize(x1)
        x2 = self.conv2(x1, edge_index, flatten_weights)
        x2 = self.normalize(x2)
        x3 = self.conv3(x2, edge_index, flatten_weights)
        x3 = self.normalize(x3)
        x4 = self.conv4(x3, edge_index, flatten_weights)
        x4 = self.normalize(x4)
        x5 = torch.mean(x4, dim=-2)
        x6 = self.l1(x5)
        nn.ReLU(inplace=True)(x6)
        x7 = self.l2(x6)

        return x7

class GAT23Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_out, dropout=dp_rate)
        self.b1 = BatchNorm(c_out)
        self.conv2 = GAT(c_out, c_hidden, concat=False, dropout=dp_rate)
        self.b2 = BatchNorm(c_hidden)
        self.conv3 = GAT(c_hidden, c_out, concat=False, dropout=dp_rate)
        self.b3 = BatchNorm(c_out)
        self.conv4 = GAT(c_out, 1, edge_dim=edge_dim, concat=False)

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.b1(self.conv1(features, edge_index))
        x2 = self.b2(self.conv2(x1, edge_index))
        x3 = self.b3(self.conv3(x2, edge_index))
        x4 = self.conv4(x3, edge_index)
        x5 = torch.mean(x4, dim=-2)

        return x5

class GAT24Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.b1 = BatchNorm(c_hidden)
        self.conv2 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.b2 = BatchNorm(c_hidden)
        self.conv3 = GAT(c_hidden, c_out, edge_dim=edge_dim, concat=False)
        self.b3 = BatchNorm(c_out)
        self.conv4 = GAT(c_out, 1, edge_dim=edge_dim, concat=False)

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.b1(self.conv1(features, edge_index, flatten_weights))
        x2 = self.b2(self.conv2(x1, edge_index, flatten_weights))
        x3 = self.b3(self.conv3(x2, edge_index, flatten_weights))
        x4 = self.conv4(x3, edge_index, flatten_weights)
        x5 = torch.mean(x4, dim=-2)

        return x5

class GAT25Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.b1 = BatchNorm(c_hidden)
        self.conv2 = GAT(c_hidden, 1, edge_dim=edge_dim, concat=False)

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.b1(self.conv1(features, edge_index))
        x2 = self.conv2(x1, edge_index)
        x3 = torch.mean(x2, dim=-2)

        return x3

class GAT26Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        GAT = geom_nn.GATv2Conv
        self.device = device
        self.conv1 = GAT(c_in, c_hidden, edge_dim=edge_dim)
        self.b1 = BatchNorm(c_hidden)
        self.conv2 = GAT(c_hidden, c_hidden, edge_dim=edge_dim, concat=False)
        self.b2 = BatchNorm(c_hidden)
        self.conv3 = GAT(c_hidden, c_out, edge_dim=edge_dim, concat=False)
        self.b3 = BatchNorm(c_out)
        self.conv4 = GAT(c_out, 1, edge_dim=edge_dim, concat=False)

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        flatten_weights_array = edge_weights.squeeze()[edge_weights.squeeze() > 1 / threashold]
        edge_index_array = np.argwhere(edge_weights.squeeze() > 1 / threashold)
        features_array = features.squeeze().double()

        flatten_weights = torch.tensor(flatten_weights_array, device=self.device).double()
        edge_index = torch.tensor(edge_index_array, device=self.device)
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.b1(self.conv1(features, edge_index))
        x2 = self.b2(self.conv2(x1, edge_index))
        x3 = self.b3(self.conv3(x2, edge_index))
        x4 = self.conv4(x3, edge_index)
        x5 = torch.mean(x4, dim=-2)

        return x5

class CONV0Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        self.device = device
        s1 = 24 * c_in
        #self.c1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, dilation=3)
        self.l1 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(s1, 1)
        )

    @staticmethod
    def calculate_conv_size(in_channels, kernel_size, dilation, stride=1):
        return int((in_channels - dilation * (kernel_size - 1) - 1) / stride) + 1

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        features_array = features.squeeze().double()
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.l1(features.flatten())

        return x1

class CONV1Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        self.device = device
        s1 = 24 * c_in
        #self.c1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, dilation=3)
        self.l1 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(s1, int(s1 / 2)),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Linear(int(s1 / 2), 1)

    @staticmethod
    def calculate_conv_size(in_channels, kernel_size, dilation, stride=1):
        return int((in_channels - dilation * (kernel_size - 1) - 1) / stride) + 1

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        features_array = features.squeeze().double()
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.l1(features.flatten())
        x2 = self.l2(x1)

        return x2

class CONV2Model(nn.Module):

    NUM_AMINOS = 24

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        self.device = device
        self.c1 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Conv1d(in_channels=c_in, out_channels=int(c_in/2), kernel_size=5),
            nn.ReLU(inplace=True)
        )
        channel_size = CONV2Model.calculate_conv_size(in_features=CONV2Model.NUM_AMINOS, kernel_size=5, dilation=1)

        self.c2 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Conv1d(in_channels=int(c_in / 2), out_channels=int(c_in/4), kernel_size=5),
            nn.ReLU(inplace=True)
        )
        channel_size = CONV2Model.calculate_conv_size(in_features=channel_size, kernel_size=5, dilation=1)

        self.l1 = nn.Sequential(
            nn.Linear(channel_size * int(c_in / 4), 1),
        )

    @staticmethod
    def calculate_conv_size(in_features, kernel_size, dilation, stride=1):
        return int((in_features - dilation * (kernel_size - 1) - 1) / stride) + 1

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        features_array = features.squeeze().double()
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.c1(features.T)
        x1 = x1 / (torch.sum(x1) + 0.1)
        x2 = self.c2(x1)
        x2 = x2 / (torch.sum(x2) + 0.1)
        x3 = self.l1(x2.flatten())

        return x3

class CONV3Model(nn.Module):

    NUM_AMINOS = 24

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        self.device = device
        self.c1 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Conv1d(in_channels=c_in, out_channels=int(c_in/2), kernel_size=5),
            nn.ReLU(inplace=True)
        )
        channel_size = CONV3Model.calculate_conv_size(in_features=CONV2Model.NUM_AMINOS, kernel_size=5, dilation=1)

        self.c2 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Conv1d(in_channels=int(c_in / 2), out_channels=int(c_in/4), kernel_size=5),
            nn.ReLU(inplace=True)
        )
        channel_size = CONV3Model.calculate_conv_size(in_features=channel_size, kernel_size=5, dilation=1)

        self.l1 = nn.Sequential(
            nn.Linear(channel_size * int(c_in / 4), 1),
        )

    @staticmethod
    def calculate_conv_size(in_features, kernel_size, dilation, stride=1):
        return int((in_features - dilation * (kernel_size - 1) - 1) / stride) + 1

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        features_array = features.squeeze().double()
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.c1(features.T)
        x1 = x1 
        x2 = self.c2(x1)
        x2 = x2
        x3 = self.l1(x2.flatten())

        return x3

class CONV4Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        self.device = device
        s1 = c_in
        #self.c1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, dilation=3)
        self.l1 = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(s1, int(s1 / 2)),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Linear(int(s1 / 2), 1)

    @staticmethod
    def calculate_conv_size(in_channels, kernel_size, dilation, stride=1):
        return int((in_channels - dilation * (kernel_size - 1) - 1) / stride) + 1

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        features_array = features.squeeze().double()
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.l1(features.flatten())
        x2 = self.l2(x1)

        return x2


class CONV5Model(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        self.device = device
        s0 = c_in
        self.c1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, dilation=3),
            nn.ReLU(inplace=True)
        )
        s1 = self.calculate_conv_size(s0, kernel_size=10, dilation=3)
        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, dilation=3),
            nn.ReLU(inplace=True)
        )
        s2 = self.calculate_conv_size(s1, kernel_size=10, dilation=3)
        self.c3 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, dilation=3),
            nn.ReLU(inplace=True)
        )
        s3 = self.calculate_conv_size(s2, kernel_size=10, dilation=3)
        self.c4 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, dilation=3),
            nn.ReLU(inplace=True)
        )
        s4 = self.calculate_conv_size(s3, kernel_size=10, dilation=3)
        self.l1 = nn.Sequential(
            nn.Linear(s4, 1),
        )

    @staticmethod
    def calculate_conv_size(in_channels, kernel_size, dilation, stride=1):
        return int((in_channels - dilation * (kernel_size - 1) - 1) / stride) + 1

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        features_array = features.squeeze().double()
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.c1(features.flatten().unsqueeze(dim=0))
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        x5 = self.l1(x4)

        return x5


class CONV6Model(nn.Module):

    NUM_AMINOS = 24

    def __init__(self, c_in, c_hidden, c_out, device, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        self.device = device

        self.l1 = nn.Sequential(
            nn.Linear(self.NUM_AMINOS, int(self.NUM_AMINOS / 2)),
        )
        self.l2 = nn.Sequential(
            nn.Linear(int(self.NUM_AMINOS / 2), 1),
        )
        self.l3 = nn.Sequential(
            nn.Linear(c_in, 1),
        )

    @staticmethod
    def calculate_conv_size(in_features, kernel_size, dilation, stride=1):
        return int((in_features - dilation * (kernel_size - 1) - 1) / stride) + 1

    @staticmethod
    def normalize(x):
        return (x - torch.mean(x, dim=-1).unsqueeze(-1)) / torch.std(x, dim=-1).unsqueeze(-1)

    def forward(self, features, edge_weights, threashold=0):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        features_array = features.squeeze().double()
        features = torch.tensor(features_array, device=self.device).double()

        x1 = self.l1(features.T)
        x2 = self.l2(x1)
        x3 = self.l3(x2.T)

        return x3
