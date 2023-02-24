import pandas as pd
import os
import numpy as np
from Bio import PDB
from scipy.spatial import distance
import time

CUTTED_PARTS_FOLER = "/cs/labs/dina/meitar/rhodopsins/cutted_parts/"
CUTTED_PARTS_PATTERN = "cutted_parts{}.pdb"
SOURCE_FOLDER = "/cs/labs/dina/meitar/rhodopsins/retinas/"
STATS_FOLDER = "/cs/labs/dina/meitar/rhodopsins/retina_alignment_files/"
RMSD_SCRIPT = "/cs/staff/dina/utils/rmsd"
ALIGN_SCRIPT = "/cs/labs/dina/meitar/rhodopsins/scripts/align_from_retina_stats.pl"
NEW_PDBS_FOLDER = "/cs/labs/dina/meitar/rhodopsins/aligned_cutted_parts/"
SOURCE_PATTERN = "match_{}[1].{}"
DEST_PATTERN = "match_{}.{}"
ALIGN_TO_NUMBER = 0
SIZE = 884

def get_points_from_lines(lines):
    points_list = [line.split()[-6:-3] for line in lines]
    points = np.array(points_list, dtype=np.float)
    return points

def compute_transofrmation(input_file_lines, target_file_lines):
    points1 = get_points_from_lines(input_file_lines)
    points2 = get_points_from_lines(target_file_lines)
    assert points1.shape[0] == 20
    assert points2.shape[0] == 20

    cent1 = np.mean(points1, axis=0)
    cent2 = np.mean(points2, axis=0)

    H = (points1 - cent1).T @ (points2 - cent2)
    U, _, Vt = np.linalg.svd(H)
    R = (U @ Vt)

    if np.linalg.det(R) < 0:
        Vt[2] *= -1
        R = (U @ Vt)


    t = cent2 - cent1 @ R
    arr = (points1 @ R + t) - points2
    print("rmsd is ", np.linalg.norm(arr) / np.sqrt(arr.shape[0]))

    return R, t


def generate_file_names(index):
    input_file = SOURCE_FOLDER + SOURCE_PATTERN.format(index, "pdb")
    alignment_file = SOURCE_FOLDER + SOURCE_PATTERN.format(ALIGN_TO_NUMBER, "pdb")
    stats_file = STATS_FOLDER + DEST_PATTERN.format(index, "stats")
    cutted_part_input_file = CUTTED_PARTS_FOLER + CUTTED_PARTS_PATTERN.format(index)
    cutted_part_target_file = CUTTED_PARTS_FOLER + CUTTED_PARTS_PATTERN.format(ALIGN_TO_NUMBER)

    return input_file, alignment_file, stats_file, cutted_part_input_file, cutted_part_target_file

def my_compute_transformation(input_file, alignment_file, target_file):
    if not os.path.isfile(input_file):
        print("Skip ", input_file)
        return
    with open(input_file, "r") as f:
        input_lines = f.readlines()[1:-2]
    with open(alignment_file, "r") as f:
        target_lines = f.readlines()[1:-2]


    compute_transofrmation(input_lines, target_lines)

def dina_compute_transformation(input_file, alignment_file, target_file, cutted_part_input, cutted_part_target):
    cmd = f"({RMSD_SCRIPT} -t {alignment_file} {input_file} | head -n1) > {target_file}"
    os.system(cmd)

    cmd = f"echo {cutted_part_target} >> {target_file}"
    os.system(cmd)

    cmd = f"echo {cutted_part_input} >> {target_file}"
    os.system(cmd)

def generate_stats_files():
    for index in range(0, SIZE):
        input_file, alignment_file, stats_file, cutted_part_input, cutted_part_target = generate_file_names(index)
        if not os.path.isfile(input_file):
            print("Skip ", input_file)
            continue
        # my_compute_transformation(input_file, alignment_file, target_file)
        dina_compute_transformation(input_file, alignment_file, stats_file, cutted_part_input, cutted_part_target)

def generate_pdbs():
    for index in range(0, SIZE):
        _, _ , stats_file, _, _ = generate_file_names(index)
        new_pdb = NEW_PDBS_FOLDER + CUTTED_PARTS_PATTERN.format(index)
        os.system(f"{ALIGN_SCRIPT} {stats_file} {new_pdb}")

def main():
    """
    generate_distance_matrices_from_folder("/cs/labs/dina/meitar/rhodopsins/pdbs/",
                                           "/cs/labs/dina/meitar/rhodopsins/graphs/")
    """
    generate_stats_files()
    generate_pdbs()

if __name__ == "__main__":
    main()
