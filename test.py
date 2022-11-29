import torch
import numpy as np
import torch.nn as nn
from torch import optim
import pickle
from torch.utils.data import DataLoader
from itertools import product
import pickle
import numpy as np

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
    dataset = PDBDataset(PDBDatasetConfig())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print("starting loop")
    loss_sum = 0
    index = 0
    result_losses = []
    while True:
        for i, graph in enumerate(dataloader):
            if i % 18 != 0:
                continue
            index += 1
            if graph[2] == 0:
                print("skipping")
                continue

            result = model.double().forward(graph[0], graph[1], 10)
            loss = torch.norm(result - graph[2].to(device))
            loss_sum += loss.item()
            result_losses.append(loss.item())
            print("losses: " + str(result_losses))
            print("current mean: " + str(np.mean(result_losses)))
            print("current std: " + str(np.std(result_losses)))

            if loss.item() > 1000:
                print("error on", graph[2].item())


if __name__ == "__main__":
    main()
