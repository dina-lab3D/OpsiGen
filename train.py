import torch
import numpy as np
import torch.nn as nn
from torch import optim
import pickle
from torch.utils.data import DataLoader
from itertools import product
import pickle
import matplotlib.pyplot as plt
import os

import models
from pdb_parser import ELEMENTS
from pdb_dataset import PDBDataset, PDBDatasetConfig
import argparse
import wandb

FILE_PATH = '/mnt/c/Users/zlils/Documents/university/biology/rhodopsins/excel/data.xlsx'
TRAIN_FILES = '/cs/labs/dina/meitar/rhodopsins/splits/train0'
os.environ["WANDB_API_KEY"] = "5ac206c34b33ff51b16fb8dcdb2efaa69943e237"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('pickle_file')
    parser.add_argument('wildtypes_list')
    args = parser.parse_args()

    return args

def set_config(args):
    config = dict()
    config["pickle_file"] = args.pickle_file
    config["wildtyes_list"] = args.wildtypes_list
    config["lr"] = 0.0001
    config["hidden_layer_size"] = 2000
    config["out_layer_size"] = 2000
    config["graph_th"] = 5

    return config


def main():
    print("Hello!")
    args = parse_arguments()
    config = set_config(args)
    wandb.init(config=config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    to_load = False
    if to_load:
        with open(args.pickle_file, 'rb') as f:
            model = pickle.load(f).to(device)
    else:
        print(config)
        model = models.GATModel(18, config["hidden_layer_size"], config["out_layer_size"], device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    dataset = PDBDataset(PDBDatasetConfig(), args.wildtypes_list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print("starting loop")
    losses_graph = []
    loss_sum = 0
    loss_train = []
    index = 0
    EV = 1239.8
    while True:
        for i, graph in enumerate(dataloader):
            if index % 100 == 0:
                with open(args.pickle_file, 'wb') as f:
                    pickle.dump(model, f)
            if graph[2] == 0:
                continue 
            index += 1
            optimizer.zero_grad()
            result = model.double().forward(graph[0], graph[1], config["graph_th"])
            if result.item() < EV and result.item() > 1:
                loss = torch.norm((EV / result) - (EV / graph[2]).to(device)) * 100
            else:
                loss = torch.norm(result - graph[2].to(device)) * 100
            wandb.log({"loss": loss.item(), "epoch": index})
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    main()
