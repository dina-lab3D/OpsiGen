import torch
import numpy as np
import torch.nn as nn
from torch import optim
import pickle
from torch.utils.data import DataLoader
from itertools import product

import models
from pdb_parser import ELEMENTS
from pdb_dataset import PDBDataset, PDBDatasetConfig


FILE_PATH = '/mnt/c/Users/zlils/Documents/university/biology/rhodopsins/excel/data.xlsx'


def main():
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.GATModel(18, 50, 1)
    loss_module = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    dataset = PDBDataset(PDBDatasetConfig())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for graph in dataloader:
        result = model.forward(graph[0], graph[1], graph[2])
        print(result)


if __name__ == "__main__":
    main()