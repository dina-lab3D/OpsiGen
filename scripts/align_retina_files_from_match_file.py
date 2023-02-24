import os
import numpy

MATCHING_FILES = "/cs/labs/dina/meitar/rhodopsins/matches/"
RESULT_DIR = "/cs/labs/dina/meitar/rhodopsins/aligned_pdbs/"
RESULT_STRING = "result"
RMSD_INDEX = 13

def parse_line(line):
    tokens = line.split()
    print(tokens[RMSD_INDEX])
    

def matching_file_to_alignment(matching_file):
    file_name = matching_file.split('/')[-1]
    file_name = file_name.replace('.stats', '.pdb')
    os.system("/cs/labs/dina/meitar/rhodopsins/scripts/align_from_stats.pl {} {}".format(matching_file, RESULT_DIR + file_name))
    with open(matching_file, "r+") as f:
        for line in f.readlines():
            if RESULT_STRING in line:
                parse_line(line)

def align_dir_recursively(dirpath):
    for root, dirs, files in os.walk(dirpath):
        for filename in files:
            matching_file_to_alignment(os.path.join(root, filename))
        for dirname in dirs:
            align_dir_recursively(os.path.join(root, dirname))

def main():
    align_dir_recursively(MATCHING_FILES)


if __name__ == "__main__":
    main()
