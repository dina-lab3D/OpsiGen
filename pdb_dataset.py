from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import excel_parser
import os
from dataclasses import dataclass


FILE_PATH = './excel/data.xlsx'
WILDTYPES_LIST = '/cs/labs/dina/meitar/rhodopsins/splits/train0'


# a simple custom collate function, just to show the idea

@dataclass
class PDBDatasetConfig:
    excel_path = '/cs/labs/dina/meitar/rhodopsins/excel/data.xlsx'
    graph_dists_path = '/cs/labs/dina/meitar/rhodopsins/graphs/'
    graph_features_path = '/cs/labs/dina/meitar/rhodopsins/features/'


class PDBDataset(Dataset):
    def __init__(self, config, wildtypes_file):
        self.config = config
        self.excel_data = pd.read_excel(config.excel_path)
        with open(wildtypes_file, "r") as f:
            self.wildtypes_names = [line[:-1] for line in f.readlines()]

    def get_category(self, category):
        return list(self.excel_data[category])

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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        print(batch)


if __name__ == "__main__":
    main()
