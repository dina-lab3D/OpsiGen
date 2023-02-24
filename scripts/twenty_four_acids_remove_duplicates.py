import os

PDBS_DIR = "/cs/labs/dina/meitar/rhodopsins/pdbs"
RETINAS_DIR = "/cs/labs/dina/meitar/rhodopsins/their_cutted_parts"
RESULT_DIR = "/cs/labs/dina/meitar/rhodopsins/their_cutted_parts2"
THREASHOLD = 2


CMD = "timeout 10s ~dina/utils/srcs/interface/interface {} {}/cutted_parts{}.pdb {} -p {}/cutted_parts_tmp{}.pdb"

def get_file_name(idx):
    for dirpath, _, filenames in os.walk(PDBS_DIR):
        for file_name in filenames:
            if "_{}_unrelaxed_rank_1_model".format(idx) in file_name:
                return os.path.join(dirpath, file_name)


def main():
    for i in range(884):
        pdb_filename = get_file_name(i)
        os.system(CMD.format(pdb_filename, RETINAS_DIR, i, THREASHOLD, RESULT_DIR, i))

    for i in range(884):
        source_file = "{}/cutted_parts{}.pdb".format(RETINAS_DIR, i)
        tmp_file = "{}/cutted_parts_tmp{}.pdb".format(RESULT_DIR, i)
        dest_file = "{}/cutted_parts{}.pdb".format(RESULT_DIR, i)
        with open(source_file, "r") as f:
            source_file_size = len(f.readlines())
        with open(dest_file, "r") as f:
            tmp_file_size = len(f.readlines())

        os.system("head {} -n {} > {}".format(tmp_file, tmp_file_size - source_file_size, dest_file))
        os.system("rm -rf {}".format(tmp_file))


if __name__ == "__main__":
    main()
