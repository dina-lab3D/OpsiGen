import numpy as np
import argparse
import os
from protein import Protein


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("original_pdbs")
    parser.add_argument("atoms_feature_dir")
    parser.add_argument("new_features_dir")

    return parser.parse_args()

def get_file_names(input_dir):

    result = []
    
    for dirpath, _, filenames in os.walk(input_dir):
        for filename in filenames:
            no_suffix = "".join(filename.split(".")[:-1])
            result.append(no_suffix)

    result.sort()
    print(result)

    return result

def loop_through_proteins(input_pdb_dir, input_feature_dir, output_feature_dir):
    PDB_SUFFIX = ".pdb"
    FEATURE_SUFFIX = ".npz"
    input_pdbs = get_file_names(input_pdb_dir)
    input_features = get_file_names(input_feature_dir)

    assert input_pdbs == input_features

    for i, file_name in enumerate(input_pdbs):
        try:
            pdb_path = os.path.join(input_pdb_dir, file_name + PDB_SUFFIX)
            last_feature_path = os.path.join(input_feature_dir, file_name + FEATURE_SUFFIX)
            new_feature_path = os.path.join(output_feature_dir, file_name + FEATURE_SUFFIX)
            curr_pro = Protein(pdb_path, last_feature_path, i)
            curr_pro.save_new_features(new_feature_path)
        except FileNotFoundError:
            print(file_name.format(i, "pdb"), "not found")


def main():
    args = parse_args()
    loop_through_proteins(args.original_pdbs, args.atoms_feature_dir, args.new_features_dir)


if __name__ == "__main__":
    main()
