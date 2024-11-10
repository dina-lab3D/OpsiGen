import os
import cutting.main as preprocess
import subprocess
import argparse
import shutil
import prediction.calculate_one_rhodopsin_4mean as predict
import prediction.models as models
from cutting.all_configs import Configs


def run_rhomax(pdb_path):
    shutil.copy(pdb_path, "./cutting/cur.pdb")
    os.chdir("./cutting/")  # change when applying
    preprocess.main(Configs.all["config"])
    os.chdir("../prediction/")
    predict.main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_path')
    args = parser.parse_args()
    run_rhomax(args.pdb_path)
