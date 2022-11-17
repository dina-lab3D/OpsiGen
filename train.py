import torch
import numpy as np
import torch.nn as nn
from torch import optim
import pickle
from torch.utils.data import DataLoader
from itertools import product
import pickle

import models
from pdb_parser import ELEMENTS
from pdb_dataset import PDBDataset, PDBDatasetConfig


FILE_PATH = '/mnt/c/Users/zlils/Documents/university/biology/rhodopsins/excel/data.xlsx'


def main():
    print("Hello!")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    to_load = True
    if to_load:
        with open('model.pckl', 'rb') as f:
            model = pickle.load(f).to(device)
    else:
        model = models.GATModel(18, 15, 1, device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1)
    dataset = PDBDataset(PDBDatasetConfig())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print("starting loop")
    index = 0
    for graph in dataloader:
        index += 1
        if graph[2] == 0:
            continue
        if index % 15 == 0:
            with open('model.pckl', 'wb') as f:
                pickle.dump(model, f)
            print("saving")
        else:
            optimizer.zero_grad()
            result = model.double().forward(graph[0], graph[1], 6)
            loss = torch.norm(result - graph[2])
            print(result, graph[2])
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
