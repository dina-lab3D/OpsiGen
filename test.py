import torch
import numpy as np
import torch.nn as nn
from torch import optim
import pickle
from torch.utils.data import DataLoader
from itertools import product
import pickle
import matplotlib.pyplot as plt

import models
from pdb_parser import ELEMENTS
from pdb_dataset import PDBDataset, PDBDatasetConfig
import argparse
import wandb

FILE_PATH = '/mnt/c/Users/zlils/Documents/university/biology/rhodopsins/excel/data.xlsx'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('pickle_file')
    parser.add_argument('wildtypes_list')
    args = parser.parse_args()

    return args


def main():
    print("Hello!")
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    to_load = True
    with open(args.pickle_file, 'rb') as f:
        model = pickle.load(f).to(device)
    model.eval()
    dataset = PDBDataset(PDBDatasetConfig(), args.wildtypes_list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print("starting loop")
    losses_graph = []
    loss_sum = 0
    loss_train = []
    index = 0
    EV = 1239.8
    for i, graph in enumerate(dataloader):
        if graph[2] == 0:
            continue
        index += 1
        result = model.double().forward(graph[0], graph[1], 5)
        if result.item() < EV and result.item() > 1:
            loss = torch.norm((EV / result) - (EV / graph[2]).to(device)) * 100
        else:
            loss = torch.norm(result - graph[2].to(device)) * 100
        loss_train.append(loss.item())
        print((result.item()), graph[2].item(),loss.item())

if __name__ == "__main__":
    main()
