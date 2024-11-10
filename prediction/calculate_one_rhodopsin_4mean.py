import torch
import numpy as np
import torch.nn as nn
from torch import optim
import pickle
from torch.utils.data import DataLoader, WeightedRandomSampler
from itertools import product
# import prediction.models as models
import sys
import io
import importlib
import pickle
import os
import json
import time
import glob
from .pdb_dataset import PDBDataset, PDBDatasetConfig
import argparse

def calculate_l1_reg(model):
    return 0.0001 * sum(torch.norm(p, 1) for p in model.parameters())

def get_pickel_name(base_name, test_error, start_time):
    result = base_name + "_" + "YAY" + "_" + str("{0:.1f}".format(test_error)) + "_" + start_time
    return result


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def calculate_energy_loss(pred, gt):
    EV = 1239.8

    if (pred < 3 or gt < 3):
        return 7

    return (EV / pred) - (EV / gt)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("pickle_folder")
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
    config = set_config("./conf_all")
    # file_pattern = os.path.join(config["pickles_folder"], '*')
    # file_list = glob.glob(file_pattern)
    file_list = [f"{config['pickles_folder']}model{i}_final" for i in range(4)]
    results = []
    for i, file_path in enumerate(file_list):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        to_load = False
        config = set_config(f"./conf_{i}")
        # models = importlib.import_module("prediction.models")
        with open(file_path, 'rb') as f:
            model = CPU_Unpickler(f).load().to(device)
        model = model.to('cpu')

        # Verify the device
        print(next(model.parameters()).device)
        model.device = device
        print(next(model.parameters()).device)
        train_dataset = PDBDataset(generate_dataset_config(config), config["train_wildtypes_list"],
                                   normalize_last=config["dataset_normalize_last"])

        # train_dataset = PDBDataset(generate_dataset_config(config), config["train_wildtypes_list"],
        #                            normalize_last=config["dataset_normalize_last"])
        # means = train_dataset.means
        # stds = train_dataset.stds
        normalized_features, dists = train_dataset.get_specific_item(config["dists_file"], config["features_file"], range(36))
        result = model.double().forward(torch.tensor(normalized_features), torch.tensor(dists), config["graph_th"])
        results.append(result.item())
        print(f"predicted wavelength by current model: {results[-1]}")

    print("outputing to " + str(config["output_file"]))
    res_text = f"absorption wavelength is {str(round(sum(results)/len(results), 3))}"
    with open(config["output_file"], "w") as f:
        f.write(res_text)
    print(res_text)
        


# args.output_file
if __name__ == "__main__":
    main()
