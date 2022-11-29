import os
import numpy
from Bio.PDB import PDBParser

MATCHING_FILES = "/cs/labs/dina/meitar/rhodopsins/matches/"
ALIGNED_DIR = "/cs/labs/dina/meitar/rhodopsins/aligned_pdbs/"
RETINA_DIR = "/cs/labs/dina/meitar/rhodopsins/retina_pdbs/"
RESULT_STRING = "result"
RMSD_INDEX = 13

def get_rmsd(matching_file):
    with open(matching_file, "r+") as f:
        for line in f.readlines():
            if RESULT_STRING in line:
                tokens = line.split()
                return tokens[RMSD_INDEX]

def add_retina_to_aligned_pdb(aligned_pdb, match_pdb, result_pdb):
    with open(match_pdb, "r") as f:
        match_lines = f.readlines()
        retina_pdb = match_lines[2].strip()
    with open(aligned_pdb, "r") as f:
        pdb_lines = f.readlines()
    with open(retina_pdb, "r") as f:
        retina_lines = f.readlines()

    ret_lines = [line for line in retina_lines if (('RET' in line) and ('HETATM' in line))]

    with open(result_pdb, "w+") as f:
        for line in pdb_lines:
            if "END" in line and ("MDL" not in line):
                continue
            else:
                f.write(line)
        f.write("MODEL 2\n")
        for line in ret_lines:
            f.write(line)
        f.write("ENDMDL\n")
        f.write("END")

def matching_file_to_alignment(matching_file):
    file_name = matching_file.split('/')[-1]
    file_name = file_name.replace('.stats', '.pdb')
    os.system("/cs/labs/dina/meitar/rhodopsins/scripts/align_from_stats.pl {} {}".format(matching_file, ALIGNED_DIR + file_name))
    # print(get_rmsd(matching_file))
    add_retina_to_aligned_pdb(ALIGNED_DIR + file_name, matching_file, RETINA_DIR + file_name)

def add_retina_to_dir_recursively(dirpath):
    for root, dirs, files in os.walk(dirpath):
        for filename in files:
            matching_file_to_alignment(os.path.join(root, filename))
        for dirname in dirs:
            add_retina_to_dir_recursively(os.path.join(root, dirname))

def main():
    add_retina_to_dir_recursively(MATCHING_FILES)


if __name__ == "__main__":
    main()
