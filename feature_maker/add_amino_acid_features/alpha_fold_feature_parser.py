import numpy as np
import os
import time

FEATURE_FILE = "/cs/labs/dina/meitar/rhodopsins/new_fastas/XR_XR_795_single_repr_1_model_4.npy"
PDB_FILE = "/cs/labs/dina/meitar/rhodopsins/new_fastas/XR_XR_795_unrelaxed_rank_1_model_4.pdb"

NP_FOLDER = "/cs/labs/dina/meitar/rhodopsins/new_fastas/"
PDB_FOLDER = "/cs/labs/dina/meitar/rhodopsins/pdbs/"

AMINO_ACID_INDEX = 3
AMINO_ACID_NUMBER_INDEX = 5

class AlphaFoldFeatureMaker:

    def __init__(self, np_folder=NP_FOLDER, pdb_folder=PDB_FOLDER):
        self.np_folder = np_folder
        self.pdb_folder = pdb_folder

        for rootdir, _, file_names in os.walk(self.np_folder):
            self.npy_files = file_names
        self.npy_rootdir = rootdir

        for rootdir, _, file_names in os.walk(self.pdb_folder):
            self.pdb_files = file_names
        self.pdb_rootdir = rootdir

    def _get_npy_file(self, index):

        chosen_file = self.npy_rootdir

        for file_name in self.npy_files:
            if f"_{index}_single_repr_1" in file_name and file_name.endswith(".npy"):
                chosen_file += file_name
                break

        return chosen_file

    def _get_pdb_file(self, index):
        chosen_file = self.pdb_rootdir

        for file_name in self.pdb_files:
            if f"_{index}_unrelaxed_rank_1" in file_name and file_name.endswith(".pdb"):
                chosen_file += file_name
                break

        return chosen_file

    def get(self, index):
        pdb_file = self._get_pdb_file(index)
        np_file = self._get_npy_file(index)

        with open(pdb_file, "r") as f:
            lines = f.readlines()

        splitted_lines = [line.split() for line in lines if "ATOM" in line]
        amino_index_to_name = dict()
        for arr in splitted_lines:
            amino_index_to_name[int(arr[AMINO_ACID_NUMBER_INDEX])] = arr[AMINO_ACID_INDEX]

        feat = np.load(np_file)
        max_length = len(amino_index_to_name)
        return feat[:max_length]
