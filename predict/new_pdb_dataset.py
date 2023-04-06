from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
import excel_parser
import os
from dataclasses import dataclass
from collections import Counter


FILE_PATH = './excel/data.xlsx'
WILDTYPES_LIST = '/cs/labs/dina/meitar/rhodopsins/splits/test0'


# a simple custom collate function, just to show the idea

@dataclass
class PDBDatasetConfig:
    sequences_path = '/cs/labs/dina/meitar/rhodopsins/excel/sequences.fas'
    wavelength_path = '/cs/labs/dina/meitar/rhodopsins/excel/wavelength.dat'
    graph_dists_path = '/cs/labs/dina/meitar/rhodopsins/graphs/'
    graph_features_path = '/cs/labs/dina/meitar/rhodopsins/features/'


class PDBDataset(Dataset):
    def __init__(self, config, wildtypes_file):
        self.config = config
        with open(wildtypes_file, "r") as f:
            self.wildtypes_names = [line[:-1] for line in f.readlines()]

        self.data = self.get_data_from_config()

    def get_category(self, category):
        return list(self.excel_data[category])

    def get_data_from_config(self):
        sequences_lines = []
        wavelength_lines = []
        with open(self.config.sequences_path, "r") as f:
            sequences_lines = f.readlines()

        with open(self.config.wavelength_path, "r") as f:
            wavelength_lines = f.readlines()

        indexes = [i for i in range(len(wavelength_lines)) if wavelength_lines[i] != 'NA\n']

        wl = [int(wavelength_lines[i]) for i in indexes]
        sl = [sequences_lines[i * 2] for i in indexes]
        wildtypes = []
        for s in sl:
            token = s.split(' ')[0]
            token = token.split('_')[0]
            token = token.split('.')[0]
            wildtypes.append(token)

        print(wildtypes)

        breakpoint()

        print(wavelength_lines)
        pass


    def calculate_weights(self):
        wildtypes_counter = {}
        wildtypes = self.excel_data["Wildtype"].values.tolist()
        wildtypes_counter = Counter(wildtypes)
        weights = np.zeros(len(wildtypes))
        for i in range(weights.shape[0]):
            weights[i] = 1 / wildtypes_counter[wildtypes[i]]

        return weights

    def __len__(self):
        return self.excel_data.shape[0] - 1

    @staticmethod
    def read_graph(dists_file_name, features_file_name):
        print(dists_file_name)
        dists = None
        features = None
        if os.path.exists(dists_file_name):
            dists = np.load(dists_file_name)

        if os.path.exists(features_file_name):
            features = np.load(features_file_name)

        return features, dists

    def __getitem__(self, idx):
        entry = self.excel_data.iloc[idx]
        features, dists = PDBDataset.read_graph(self.config.graph_dists_path + "cutted_parts{}_dists.npy".format(idx), self.config.graph_features_path + "cutted_parts{}.npz".format(idx))

        wildtype = self.get_category('Wildtype')[idx]
        lmax = self.get_category('lmax')[idx]
        if dists is None or features is None:
            if dists is None:
                print("dists is None")
            else:
                print("features is None")
            return [], [], 0
        if wildtype not in self.wildtypes_names:
            return [], [], 0
        return features, dists, lmax


def main():
    dataset = PDBDataset(PDBDatasetConfig(), WILDTYPES_LIST)
    # dataset.calculate_weights()
    # print(len(dataset))


if __name__ == "__main__":
    main()
