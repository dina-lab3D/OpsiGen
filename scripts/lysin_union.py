import os

PDBS_DIR = "/cs/labs/dina/meitar/rhodopsins/aligned_pdbs"
RETINAS_DIR = "/cs/labs/dina/meitar/rhodopsins/lysins"
RESULT_DIR = "/cs/labs/dina/meitar/rhodopsins/cutted_lysin"
THREASHOLD = 10


CMD = "timeout 10s ~dina/utils/srcs/interface/interface {}/match_{}\[1].pdb {}/chosen_lys{}.pdb {} -p {}/cutted_parts{}.pdb"

def main():
    for i in range(884):
        os.system(CMD.format(PDBS_DIR, i, RETINAS_DIR, i, THREASHOLD, RESULT_DIR, i))


if __name__ == "__main__":
    main()
