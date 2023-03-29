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
    if to_load:
        with open(config["pickle_file"], 'rb') as f:
            model = pickle.load(f).to(device)
    else:
        model_factory = getattr(models, config["model_name"])
        model = model_factory(config["number_features"], config["hidden_layer_size"], config["out_layer_size"], device, dp_rate=0.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.00001)
    train_dataset = PDBDataset(generate_dataset_config(config), config["train_wildtypes_list"], normalize_last=config["dataset_normalize_last"])
    to_shuffle = False
    if to_shuffle:
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=1, sampler=WeightedRandomSampler(train_dataset.calculate_weights(), num_samples=884))

    test_dataset = PDBDataset(generate_dataset_config(config), config["test_wildtypes_list"], means=train_dataset.means, stds=train_dataset.stds, normalize_last=config["dataset_normalize_last"])
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    print("starting loop")
    losses_graph = []
    loss_sum_train = 0
    loss_sum_test = 0
    energy_loss_sum_test = 0
    loss_train = []
    index = 0
    EV = 1239.8
    train_length = 0
    test_length = 0
    while True:
        index += 1
        loss_sum_train = 0
        loss_sum_test = 0
        train_length = 0
        test_length = 0
        energy_loss_sum_test = 0
        for i, graph in enumerate(train_dataloader):
            if graph[2] == 0:
                continue 
            train_length += 1
            optimizer.zero_grad()
            result = model.double().forward(graph[0], graph[1], config["graph_th"])
            #print(result, graph[2])
            loss = torch.norm(result - graph[2].to(device)) + calculate_l1_reg(model)
            loss_sum_train += loss.item()
            loss.backward()
            optimizer.step()
        
        for i, graph in enumerate(test_dataloader):
            if graph[2] == 0:
                continue 
            test_length += 1
            result = model.double().forward(graph[0], graph[1], config["graph_th"])
            energy_loss = calculate_energy_loss(result, graph[2].to(device))
            energy_loss_sum_test += energy_loss
            loss = torch.norm(result - graph[2].to(device))
            loss_sum_test += loss.item()

        wandb.log({"train_loss": loss_sum_train / train_length, "test_loss": loss_sum_test / test_length, "energy_loss": energy_loss_sum_test / test_length})

        if loss_sum_test / test_length < config["test_goal"]:
            res_name = get_pickel_name(config["pickle_file"], loss_sum_test / test_length, t)
            with open(res_name, 'wb') as f:
                pickle.dump(model, f)

if __name__ == "__main__":
    main()
