"""
generate graph features.
Generate nodes with physio-chemical descriptors. Cut only the 24 relevant amino acids.
Generate graph edges between nodes x,y such that the edge will be 1/d(x,y).
"""
import os
from .aligner_structural import Aligner
from .cutter import Cutter
from .all_configs import Configs
from tqdm import tqdm
import sys
import builtins
import os
import shutil

CUT = "./cuts/cutted_parts0.pdb"
FEATURES = "./features/cutted_parts0.npz"
DISTS = "./dists/cutted_parts0_dists.npy"



def generate_pdb(config):
    """
    Cut the relevant amino acids from the given pdb acording to the config.
    """
    aligner_obj = Aligner(config)
    pdb_cutter = Cutter(config)
    pdb_cutter.cut_pdb(aligner_obj)


def generate_features(config):
    """
    generate graph features (nodes with descriptors, graph edges) according to the config
    """
    print("Creating atom features...")
    cmd = f"{config['feature_maker_script']} {config['cutted_parts_dir']} {config['features']}"
    os.system(cmd)
    print("Creating amino acids features...")
    cmd = f"{config['amino_acid_feature_script']} \
            {config['cutted_parts_dir']} {config['features']} {config['features']}"
    os.system(cmd)
    print("Creating graph edges...")
    cmd = f"python {config['graph_maker_script']} \
            {config['cutted_parts_dir']} {config['edge_dists_path']}"
    os.system(cmd)

    print("Done!")


def main(config):
    """
    run main function of this module
    """
    generate_pdb(config)
    generate_features(config)


if __name__ == "__main__":
    config = Configs.all["config"]
