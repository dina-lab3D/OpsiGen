import os

FIRST_PDB = "/cs/labs/dina/meitar/rhodopsins/pdbs/XR_XR_795_unrelaxed_rank_1_model_4.pdb"

SECOND_PDB = "/cs/labs/dina/meitar/rhodopsins/pdbs/XR_XR_795_unrelaxed_rank_1_model_4.pdb"

def main():
    command = "./align.pl {} {}".format(FIRST_PDB, SECOND_PDB)
    os.system(command)

if __name__ == "__main__":
    main()
