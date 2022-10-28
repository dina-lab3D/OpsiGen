import torch.nn as nn
import torch.nn.functional as F

# Torchvision
import torch_geometric.nn as geom_nn
from torch_geometric.nn.norm import BatchNorm


class MLPModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, dp_rate=0.1):
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

    def forward(self, x, *args, **kwargs):
        """
        Inputs:
            x - Input features per node
        """
        return self.layers(x)


class GATModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, dp_rate=0.1, heads1=4, heads2=4, heads3=6, edge_dim=1, **kwargs):
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
        GAT = geom_nn.GATConv
        self.conv1 = GAT(c_in, c_hidden, heads=heads1, edge_dim=edge_dim)
        self.bn1 = BatchNorm(c_hidden*heads1)
        self.conv2 = GAT(c_hidden*heads1, c_hidden, heads=heads2, edge_dim=edge_dim)
        self.bn2 = BatchNorm(c_hidden*heads2)
        self.conv3 = GAT(c_hidden*heads2, c_hidden, heads=heads3, edge_dim=edge_dim, concat=False)
        self.bn3 = BatchNorm(c_hidden)
        self.head = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(c_hidden, c_out)
        )

    def forward(self, data):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """

        x, edge_index, batch_idx, edge_weights = data.x, data.edge_index, data.batch, data.edge_attr
        x1 = self.bn1(self.conv1(x, edge_index, edge_weights))
        nn.ELU(x1, inplace=True)
        x2 = self.bn2(self.conv2(x1, edge_index, edge_weights)) + x1  # skip connection
        nn.ELU(x2, inplace=True)
        x3 = self.bn3(self.conv3(x2, edge_index, edge_weights))
        x3 = geom_nn.global_mean_pool(x3, batch_idx) # aggregate average pooling
        x4 = self.head(x3)

        return x4