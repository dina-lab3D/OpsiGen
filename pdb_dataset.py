from torch.utils.data import Dataset, DataLoader
import os
from Bio import PDB
import numpy as np
import pandas as pd
import torch
import pdb_parser
import excel_parser
from dataclasses import dataclass


FILE_PATH = './excel/data.xlsx'


# a simple custom collate function, just to show the idea
def my_collate(batch):
    breakpoint()
    data = [pdb_parser.build_graph_from_atoms(item[0]) for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


@dataclass
class PDBDatasetConfig:
    excel_path = '/cs/labs/dina/meitar/rhodopsins/excel/data.xlsx'
    graph_dists_path = '/cs/labs/dina/meitar/rhodopsins/graphs/'
    graph_features_path = '/cs/labs/dina/meitar/rhodopsins/features/'


class PDBDataset(Dataset):
    def __init__(self, config):
        # self.files = []
        self.config = config
        self.excel_data = pd.read_excel(config.excel_path)
        # for filename in os.listdir(directory):
        #     f = os.path.join(directory, filename)
        #     # checking if it is a file
        #     if os.path.isfile(f):
        #         self.files.append(f)
        # self.parser = PDB.PDBParser()

    def get_category(self, category):
        return list(self.excel_data[category])

    def __len__(self):
        return self.excel_data.shape[0] - 1

    @staticmethod
    def read_graph(dists_file_name, features_file_name):
        dists = None
        features = None
        breakpoint()
        try:
            dists = np.load(dists_file_name)
            print("dists successful - {}".format(dists_file_name))
        except FileNotFoundError:
            print("failed to parse {}".format(dists_file_name))

        try:
            features = np.load(features_file_name)
            print("dists successful - {}".format(features_file_name))
        except FileNotFoundError:
            print("failed to parse {}".format(features_file_name))
        return dists, features

    def __getitem__(self, idx):
        # name = self.get_category('Name')[idx]
        entry = self.excel_data.iloc[idx]
        dists_file_names = excel_parser.entry_to_dists_file_names(entry, idx, self.config.graph_dists_path)
        features_file_names = excel_parser.entry_to_features_file_names(entry, idx, self.config.graph_features_path)
        dists = None
        features = None
        for i in range(5):
            dists, features = PDBDataset.read_graph(dists_file_names[i], features_file_names[i])
            if (not (dists is None)) and (not (features is None)):
                break

        lmax = self.get_category('lmax')[idx]
        if dists is None or features is None:
            return [], [], 0
        return dists, features, lmax


def main():
    dataset = PDBDataset(PDBDatasetConfig())
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=my_collate)
    for batch in dataloader:
        print(batch)
        breakpoint()


if __name__ == "__main__":
    main()
