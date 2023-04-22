import numpy as np
import json
import alignment
from alpha_fold_feature_parser import AlphaFoldFeatureMaker

class Protein:

    AMINO_DICT = "/cs/labs/dina/meitar/ionet-meitar/Interface_grid/add_amino_acid_features/amino_mapping"
    SEQFILE = "/cs/labs/dina/meitar/ionet-meitar/Interface_grid/add_amino_acid_features/new_sequence.fas"
    ORIGINAL_FEATURE_LEN = 18
    AMINO_FEATURE_LEN = 24
    THEIR_FEATURE_LENGTH = 18
    ALIGNMENT = alignment.Alignment(SEQFILE)
    ATOM_TYPES = ['C', 'N', 'O', 'S']

    def __init__(self, pdb_path, np_feature_file, index):
        self.amino_mapping = dict()
        with open(Protein.AMINO_DICT, "r") as f:
            amino_mapping = json.load(f)
        for key in amino_mapping:
            self.amino_mapping[key] = np.array(amino_mapping[key])
        self.atom_types = []
        self.afm = AlphaFoldFeatureMaker()

        # self.amino_mapping["RET"] = np.zeros(Protein.AMINO_FEATURE_LEN)
        self.previous_features = np.load(np_feature_file)
        with open(pdb_path, "r") as f:
            self.pdb_lines = f.readlines()

        self.new_features = self.generate_new_features()
        self.index = index

    @staticmethod
    def get_number_from_path(pdb_path):
        return int((pdb_path.split("/")[-1].split('.')[0]).split("cutted_parts")[-1])

    def save_new_features(self, path):
        with open(path, "wb") as f:
            np.save(f, self.new_features)

    def create_location_feature(self, tokens):
        feat = np.zeros(3).astype(np.float)
        feat[0] = float(tokens[-6])
        feat[1] = float(tokens[-5])
        feat[2] = float(tokens[-4])

        return feat

    def parse_pdb_line(self, line):
        tokens = line.split()
        amino_acid_index = 0
        if len(tokens) == 11 or len(tokens) == 12 or tokens[0] == 'HETATM':
            amino_acid_index = 3
        else:
            amino_acid_index = 2

        amino_acid = tokens[amino_acid_index]
        atom_type = tokens[amino_acid_index - 1]
        feat = np.copy(self.amino_mapping[amino_acid])

        return np.hstack([feat])


    def parse_pdb_lines(self):
        arr = []
        for line in self.pdb_lines:
            if len(line.split()) < 7:
                continue
            feat = self.parse_pdb_line(line)
            arr.append(feat)

        return np.vstack(arr)

    def reshape_original_feature(self):
        old_feat = self.previous_features
        total_length = old_feat.shape[0]

        return old_feat.reshape(int(total_length / self.ORIGINAL_FEATURE_LEN), self.ORIGINAL_FEATURE_LEN)


    def generate_new_features(self):
        amino_features = self.parse_pdb_lines()
        old_features = self.reshape_original_feature()
        results = np.hstack([amino_features, old_features])
        #results = amino_features

        return results


def main():
    print("Hi")

if __name__ == "__main__":
    main()
