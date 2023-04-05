import numpy as np
import argparse
from protein import Protein


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("original_pdbs")
    parser.add_argument("atoms_feature_dir")
    parser.add_argument("new_features_dir")

    return parser.parse_args()

def loop_through_proteins(input_pdb_dir, input_feature_dir, output_feature_dir):
    file_name = "cutted_parts{}.{}"
    for i in range(0, 1):
        try:
            pdb_path = input_pdb_dir + file_name.format(i, "pdb")
            last_feature_path = input_feature_dir + file_name.format(i, "npz")
            new_feature_path = output_feature_dir + file_name.format(i, "npz")
            curr_pro = Protein(pdb_path, last_feature_path, i)
            curr_pro.save_new_features(new_feature_path)
        except FileNotFoundError:
            print(file_name.format(i, "pdb"), "not found")


def main():
    args = parse_args()
    loop_through_proteins(args.original_pdbs, args.atoms_feature_dir, args.new_features_dir)


if __name__ == "__main__":
    main()
