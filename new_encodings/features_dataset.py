from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import excel_parser
from dataclasses import dataclass


FILE_PATH = './excel/data.xlsx'


# a simple custom collate function, just to show the idea

@dataclass
class MeitarDatasetConfig:
    excel_path = '/cs/labs/dina/meitar/rhodopsins/excel/data.xlsx'
    pdb_features_path = '/cs/labs/dina/meitar/rhodopsins/new_encodings/results/'


class MeitarDataset(Dataset):
    def __init__(self, config):
        self.config = config
        # self.excel_data = pd.read_excel(config.excel_path)
        self.files = []
        for dirpath, _, file_names in os.walk(config.pdb_features_path):
            for file_name in file_names:
                self.files.append(dirpath + file_name)

    def get_category(self, category):
        return list(self.excel_data[category])

    def get_wave_length(self, name):
        start = name.index('[')
        end = name.index(']')

        return name[start+1:end]

    def __len__(self):
        return len(self.files)

    @staticmethod
    def read_graph(dists_file_name, features_file_name):
        dists = None
        features = None
        try:
            dists = np.load(dists_file_name)
        except FileNotFoundError:
            pass

        try:
            features = np.load(features_file_name)
        except FileNotFoundError:
            pass
        return features, dists

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])
        lmax = self.get_wave_length(self.files[idx])

        return arr, int(lmax)


def main():
    dataset = MeitarDataset(MeitarDatasetConfig())
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for batch in dataloader:
        print(batch)


if __name__ == "__main__":
    main()
