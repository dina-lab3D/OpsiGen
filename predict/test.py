import torch
import numpy as np
import torch.nn as nn
from torch import optim
import pickle
from torch.utils.data import DataLoader, WeightedRandomSampler
from itertools import product
import pickle
import matplotlib.pyplot as plt
import os
import json
import time

import models
from pdb_parser import ELEMENTS
from pdb_dataset import PDBDataset, PDBDatasetConfig
import argparse
import wandb

FILE_PATH = '/mnt/c/Users/zlils/Documents/university/biology/rhodopsins/excel/data.xlsx'
os.environ["WANDB_API_KEY"] = "5ac206c34b33ff51b16fb8dcdb2efaa69943e237"

def calculate_l1_reg(model):
    return 0.0001 * sum(torch.norm(p, 1) for p in model.parameters())

def get_pickel_name(base_name, test_error, start_time):
    result = base_name + "_" + "YAY" + "_" + str("{0:.1f}".format(test_error)) + "_" + start_time
    return result



def calculate_energy_loss(pred, gt):
    EV = 1239.8

    if (pred < 3 or gt < 3):
        return 7

    return (EV / pred) - (EV / gt)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("pickle_file")
    parser.add_argument("output_file")
    args = parser.parse_args()

    return args

def set_config(config_file):
    config = dict()
    with open(config_file, "r") as f:
        data = json.load(f)

    print(data)
    config = data

    return config

def generate_dataset_config(config):
    dataset_config = PDBDatasetConfig()
    dataset_config.excel_path = config["excel_path"]
    dataset_config.graph_dists_path = config["graph_dists_path"]
    dataset_config.graph_features_path = config["graph_features_path"]
    dataset_config.indexes = config["indexes_to_keep"]

    return dataset_config


def main():
    breakpoint()
    print("Hello!")
    args = parse_arguments()
    config = set_config(args.config_path)
    wandb.init(config=config)
    t = str(time.time())
    print(torch.version.cuda)
    print(torch.__version__)
    print("is available", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    to_load = False
    with open(args.pickle_file, 'rb') as f:
        model = pickle.load(f).to(device)
    train_dataset = PDBDataset(generate_dataset_config(config), config["train_wildtypes_list"], normalize_last=config["dataset_normalize_last"])
    test_dataset = PDBDataset(generate_dataset_config(config), config["test_wildtypes_list"], means=train_dataset.means, stds=train_dataset.stds, normalize_last=config["dataset_normalize_last"])
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    spectra = []
    train_spectra = []

    for i, graph in enumerate(train_dataloader):
        if graph[2] == 0:
            continue 
       
        train_spectra.append(graph[2].item())
        spectra.append(graph[2].item())

    for i, graph in enumerate(test_dataloader):
        if graph[2] == 0:
            continue 
        result = model.double().forward(graph[0], graph[1], config["graph_th"])
        loss = torch.norm(result - graph[2].to(device))
        print(graph[2].item())
        with open(args.output_file, "a") as f:
            f.write(str(loss.item()) + " ")

        with open(args.output_file + "_spctra_test", "a") as f:
            spectra.append(graph[2].item())
            f.write(str(graph[2].item()) + " ")

    with open(args.output_file + "_spectra_full", "a") as f:
        for i in spectra:
            f.write(str(i) + " ")

    with open(args.output_file + "_spectra_train", "a") as f:
        for i in train_spectra:
            f.write(str(i) + " ")

if __name__ == "__main__":
    main()
