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
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    to_load = True
    if to_load:
        with open('model.pckl', 'rb') as f:
            model = pickle.load(f).to(device)
    else:
        model = models.GATModel(18, 200, 200, device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    dataset = PDBDataset(PDBDatasetConfig())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print("starting loop")
    loss_sum = 0
    index = 0
    while True:
        for graph in dataloader:
            index += 1
            if graph[2] == 0:
                print("skipping")
                continue
            if index % 100 == 0:
                loss_sum = 0
                index = 1
                with open('model.pckl', 'wb') as f:
                    pickle.dump(model, f)
                print("saving")

            optimizer.zero_grad()
            result = model.double().forward(graph[0], graph[1], 10)
            loss = torch.norm(result - graph[2].to(device))
            loss_sum += loss.item()
            print(result.item(), graph[2].item(),loss_sum / index, result.shape)
            loss.backward()
            optimizer.step()

            if loss.item() > 1000:
                print("error on", graph[2].item())


if __name__ == "__main__":
    main()
