from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
import excel_parser
import os
from dataclasses import dataclass
from collections import Counter


FILE_PATH = './excel/data.xlsx'
WILDTYPES_LIST_TEST = '/cs/labs/dina/meitar/rhodopsins/splits/test0'
WILDTYPES_LIST_TRAIN = '/cs/labs/dina/meitar/rhodopsins/splits/train0'
NORMALIZE_LAST = True


# a simple custom collate function, just to show the idea

@dataclass
class PDBDatasetConfig:
    excel_path = '/cs/labs/dina/meitar/rhodopsins/excel/data.xlsx'
    graph_dists_path = '/cs/labs/dina/meitar/rhodopsins/their_graph/'
    graph_features_path = '/cs/labs/dina/meitar/ionet-meitar/Interface_grid/new_features/24_acids_with_atoms/'


class PDBDataset(Dataset):
    def __init__(self, config, wildtypes_file, means=None, stds=None, normalize_last=True):
        self.config = config
        self.excel_data = pd.read_excel(config.excel_path)
        with open(wildtypes_file, "r") as f:
            self.wildtypes_names = [line[:-1] for line in f.readlines()]

        self.normalize_last = normalize_last

        if (means is None and stds is None):
            self.means, self.stds = 0, 1
            self.means, self.stds = self.calculate_stats_of_train()
        else:
            self.means = means
            self.stds = stds

    def calculate_stats_of_train(self):
        values = []
        for i in range(self.__len__()):
            features, dists, lmax = self.__getitem__(i)
            if len(features) == 0:
                continue
            values.append(features)

        stacked_features = np.vstack(values)
        means = np.mean(stacked_features, axis=0)
        stds = np.std(stacked_features, axis=0)

        if not self.normalize_last:
            means[-3:] = 0
            stds[-3:] = 1

        return means, stds


    def get_category(self, category):
        return list(self.excel_data[category])

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
                pass
                # print("dists is None")
            else:
                pass
                # print("features is None")
            return [], [], 0
        if wildtype not in self.wildtypes_names:
            return [], [], 0

        if isinstance(self.stds, int) or len(dists) == 0:
            return features, dists, lmax


        normalized_features = (features[:, self.stds != 0] - self.means[self.stds != 0]) / self.stds[self.stds != 0]
        return normalized_features, dists, lmax


def main():
    train_dataset = PDBDataset(PDBDatasetConfig(), WILDTYPES_LIST_TRAIN, normalize_last=True)
    test_dataset = PDBDataset(PDBDatasetConfig(), WILDTYPES_LIST_TEST, means=train_dataset.means, stds=train_dataset.stds, normalize_last=True)

    result = []
    for feat, dists, lmax in train_dataset:
        if len(feat) != 0:
            result.append(feat) 
            #print(np.std(feat))
            #print(feat[:,-3:])

    h = np.vstack(result)

    for feat, dists, lmax in test_dataset:
        if len(feat) != 0:
            result.append(feat) 
            #print(np.std(feat))
            #print(feat[:,-3:])

    h = np.vstack(result)
        


if __name__ == "__main__":
    main()
