from Bio import PDB
import os

INPUT_DIR = "/cs/labs/dina/meitar/rhodopsins/retina_pdbs/"

def main():
    my_parser = PDB.PDBParser()
    bad_files = 0
    good_files = 0
    for (dirpath, dirnames, filenames) in os.walk(INPUT_DIR):
        for filename in filenames:
            if filename.endswith('.pdb'): 
                try:
                    my_parser.get_structure(filename, os.path.join(dirpath, filename))
                    good_files += 1
                except (ValueError, FileNotFoundError) as e:
                    print(e)
                    bad_files += 1
                    print(filename, "is bad")

            print("bad files: ", bad_files, "good files:", good_files)

if __name__ == "__main__":
    main()
