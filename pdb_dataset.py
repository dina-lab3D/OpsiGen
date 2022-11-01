from torch.utils.data import Dataset, DataLoader
import os
from Bio import PDB
import pandas as pd
import torch
import pdb_parser



# a simple custom collate function, just to show the idea
def my_collate(batch):
    data = [pdb_parser.build_graph_from_atoms(item[0]) for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


class PDBDataset(Dataset):
    def __init__(self, directory, excel_path):
        self.files = []
        self.excel_data = pd.read_excel(excel_path)
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                self.files.append(f)
        self.parser = PDB.PDBParser()

    def get_category(self, category):
        return list(self.excel_data[category])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.get_category('Name')[idx]
        lmax = self.get_category('lmax')[idx]
        struct = self.parser.get_structure(idx, self.files[idx])
        atoms = struct.get_atoms()
        return list(atoms), lmax


def main():
    dataset = PDBDataset('./pdbs', FILE_PATH)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=my_collate)
    for batch in dataloader:
        print(batch)
        breakpoint()


if __name__ == "__main__":
    main()
