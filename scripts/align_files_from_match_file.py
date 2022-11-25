import os

MATCHING_FILES = "/cs/labs/dina/meitar/rhodopsins/matches/"
RESULT_STRING = "RESULT"
RMSD_INDEX = 13

def parse_line(line):
    tokens = line.split()
    print(tokens[RMSD_INDEX])

def matching_file_to_alignment(matching_file):
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
