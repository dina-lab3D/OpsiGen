import torch
import numpy as np
import torch.nn as nn
from torch import optim
import pickle
from torch.utils.data import DataLoader
from itertools import product
import pickle

import models
from features_dataset import MeitarDataset, MeitarDatasetConfig
from torch.utils.data import random_split


FILE_PATH = '/mnt/c/Users/zlils/Documents/university/biology/rhodopsins/excel/data.xlsx'


def main():
    print("Hello!")
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    to_load = False
    if to_load:
        pass
        # with open('model.pckl', 'rb') as f:
        # model = pickle.load(f).to(device)
    else:
        model = models.SimpleModel()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    full_dataset = MeitarDataset(MeitarDatasetConfig())
    train_dataset, test_dataset = random_split(full_dataset, [int(0.95 * len(full_dataset)),len(full_dataset) - int(0.95 * len(full_dataset))])
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    print("starting loop")
    loss_sum = 0
    index = 0
    for _ in range(4):
        for arrs, lmaxs in train_dataloader:
            index += 1
            if index % 100 == 0:
                loss_sum = 0
                index = 1
                # with open('model.pckl', 'wb') as f:
                #pickle.dump(model, f)
                print("saving")

            optimizer.zero_grad()
            result = model.double().forward(arrs)
            loss = torch.norm(result - lmaxs)
            print(result - lmaxs)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
    
    losses = []
    for arrs, lmaxs in test_dataloader:
        result = model.double().forward(arrs)
        loss = torch.norm(result - lmaxs)
        losses.append(loss.item()) 
        print(result.item())

    print(losses)
    print(np.mean(losses))
    print(np.std(losses))

if __name__ == "__main__":
    main()
