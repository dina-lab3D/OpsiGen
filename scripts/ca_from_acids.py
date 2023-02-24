import os

FORMAT = "grep CA /cs/labs/dina/meitar/rhodopsins/their_cutted_parts/cutted_parts{}.pdb > /cs/labs/dina/meitar/rhodopsins/ca_amino_acids/cutted_parts{}.pdb"

def main():
    for i in range(884):
        os.system(FORMAT.format(i, i))
    pass

if __name__ == "__main__":
    main()
