import pandas as pd
import os
import sys
import numpy as np
from Bio import PDB
from scipy.spatial import distance

def file_to_distance_matrix(input_path, output_path, parser, threas):
    try:
        struct = parser.get_structure(input_path, input_path)
        atoms = struct.get_atoms()
        my_atoms = [atom for atom in atoms]
        coords = np.array([atom.coord for atom in my_atoms]).astype(np.float32)
        if len(coords) == 0:
            print(input_path)
        dists = 1 / distance.cdist(coords, coords)
        dists[dists == np.inf] = 0
        np.save(output_path, dists)
    except Exception as e:
        print(e)
        breakpoint()


def generate_distance_matrices_from_folder(input_path, output_path):
    parser = PDB.PDBParser()
    for (dirpath, dirnames, filenames) in os.walk(input_path):
        for filename in filenames:
            pdb_path = os.path.join(dirpath, filename)
            npy_path = os.path.join(output_path, (filename).replace('.pdb', '_dists.npy'))
            file_to_distance_matrix(pdb_path, npy_path, parser, 16)
            # print(pdb_path, npz_path)

def main():
    generate_distance_matrices_from_folder(sys.argv[1], sys.argv[2])
    # generate_distance_matrices_from_folder("/cs/labs/dina/meitar/rhodopsins/retina_pdbs/", "/cs/labs/dina/meitar/rhodopsins/graphs/")
    #df = pd.read_excel(EXCEL_PATH)
    #generate_fasta_files(df, FASTAS_PATH)

if __name__ == "__main__":
    main()
