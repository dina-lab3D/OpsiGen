from config import parse_args
import sequences
import aligner
import cutter
import os

def generate_pdb(config):
    aligner_obj = aligner.Aligner(config)
    pdb_cutter = cutter.Cutter(config)
    pdb_cutter.cut_pdb(aligner_obj)

def generate_features(config):
    print("Creating atom features...")
    cmd = "{} {} {}".format(config["feature_maker_script"], config["cutted_parts_dir"], config["features"])
    os.system(cmd)
    print("Creating amino acids features...")
    cmd = "{} {} {} {}".format(config["amino_acid_feature_script"], config["cutted_parts_dir"], config["features"], config["features"])
    os.system(cmd)
    print("Creating graph edges...")
    cmd = "python {} {} {}".format(config["graph_maker_script"], config["cutted_parts_dir"], config["edge_dists_path"])
    os.system(cmd)

    print("Done!")


def main():
    config = parse_args()
    generate_pdb(config)
    generate_features(config)


if __name__ == "__main__":
    main()
