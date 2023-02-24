import os

PDBS_DIR = "/cs/labs/dina/meitar/rhodopsins/pdbs"
RETINAS_DIR = "/cs/labs/dina/meitar/rhodopsins/their_cutted_parts"
RESULT_DIR = "/cs/labs/dina/meitar/rhodopsins/their_cutted_parts2"
THREASHOLD = 2


CMD = "timeout 10s ~dina/utils/srcs/interface/interface {} {}/cutted_parts{}.pdb {} -p {}/cutted_parts{}.pdb"

def get_file_name(idx):
    for dirpath, _, filenames in os.walk(PDBS_DIR):
        for file_name in filenames:
            if "_{}_unrelaxed_rank_1_model".format(idx) in file_name:
                return os.path.join(dirpath, file_name)


def main():
    for i in range(884):
        filename = get_file_name(i)
        os.system(CMD.format(filename, RETINAS_DIR, i, THREASHOLD, RESULT_DIR, i))


if __name__ == "__main__":
    main()
