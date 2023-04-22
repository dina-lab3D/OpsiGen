import torch
import numpy as np
import torch.nn as nn
from torch import optim
import pickle
from torch.utils.data import DataLoader, WeightedRandomSampler
from itertools import product
import pickle
import os
import json
import time
from pdb_dataset import PDBDataset, PDBDatasetConfig
import argparse

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
    parser.add_argument("features_file")
    parser.add_argument("dists_file")
    args = parser.parse_args()

    print(args)

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
    t = str(time.time())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    to_load = False
    with open(args.pickle_file, 'rb') as f:
        model = pickle.load(f).to(device)

    means = np.load("means.npy")
    stds = np.load("stds.npy")
    train_dataset = PDBDataset(generate_dataset_config(config), config["train_wildtypes_list"], normalize_last=config["dataset_normalize_last"], means=means, stds=stds)

    normalized_features, dists = train_dataset.get_specific_item(args.dists_file, args.features_file, range(36))

    result = model.double().forward(torch.tensor(normalized_features), torch.tensor(dists), config["graph_th"])
    print("outputing to" + str(args.output_file))
    with open(args.output_file, "w") as f:
        f.write("absorption wavelength is " + str(result.item()))

if __name__ == "__main__":
    main()
