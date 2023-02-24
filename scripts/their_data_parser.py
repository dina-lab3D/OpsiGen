import pandas as pd
import os
import numpy as np
from Bio import PDB
from scipy.spatial import distance


EXCEL_PATH = "/cs/labs/dina/meitar/rhodopsins/excel/data.xlsx"
SEQUENCES_PATH = "/cs/labs/dina/meitar/rhodopsins/excel/sequences.fas"

FASTAS_TARGET_PATH = "/cs/labs/dina/meitar/rhodopsins/new_fastas/"


def generate_fasta_name(idx):
    return "seq_" + str(idx) + ".fasta"


def entry_to_dists_file_names(entry, idx, path):
    return 
    pass


def entry_to_features_file_names(entry, idx, path):
    return result


def entry_to_fasta(lines, idx, target_path):
    first_line = '>' + str(idx)
    sequence = lines[idx * 2 + 1].strip().replace('-','')

    fasta_content = first_line + '\n' + sequence
    fasta_name = generate_fasta_name(idx)

    fasta_path = os.path.join(target_path, fasta_name)
    with open(fasta_path, 'w') as f:
        f.write(fasta_content)


def entry_to_pdb_name(entry, id, path):
    return "my name"


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
        # dists[dists < threas] = -1
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


def generate_fasta_files(sequences_path, target_path):
    with open(sequences_path, "r") as f:
        lines = f.readlines()
    
    num_rhodopsins = int(len(lines) / 2)
    for i in range(num_rhodopsins):
        entry_to_fasta(lines, i, target_path)
    # for i in range(len(df)):
    # entry_to_fasta(df.iloc[i], i, path)


def main():
    # generate_distance_matrices_from_folder("/cs/labs/dina/meitar/rhodopsins/cutted_lysin/", "/cs/labs/dina/meitar/rhodopsins/lysin_graph/")
    # generate_distance_matrices_from_folder("/cs/labs/dina/meitar/rhodopsins/retina_pdbs/", "/cs/labs/dina/meitar/rhodopsins/graphs/")
    #df = pd.read_excel(EXCEL_PATH)
    generate_fasta_files(SEQUENCES_PATH, FASTAS_TARGET_PATH)

if __name__ == "__main__":
    main()
