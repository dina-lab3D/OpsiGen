import pandas as pd
import os
import numpy as np
from Bio import PDB
from scipy.spatial import distance


EXCEL_PATH = "/cs/labs/dina/meitar/rhodopsins/excel/data.xlsx"

FASTAS_PATH = "/cs/labs/dina/meitar/rhodopsins/fastas/"


def generate_fasta_name(entry, id, path):
    return str(id) + '-' + entry['Name'].replace('/','.').replace(' ', '_') + '.fasta'


def entry_to_dists_file_names(entry, id, path):
    dist_file_name_format = entry['Name'].replace('/', '-').replace(' ', '') + '_' + entry['Wildtype'] + '_' + str(id) + '_' + 'unrelaxed_rank_{}_model_{}_dists.npy'
    result = []
    for i in range(1,6):
        for j in range(1,6):
            result.append(path + dist_file_name_format.format(i, j))
    return result


def entry_to_features_file_names(entry, id, path):
    features_file_names_format = entry['Name'].replace('/', '-').replace(' ', '') + '_' + entry['Wildtype'] + '_' + str(
        id) + '_' + 'unrelaxed_rank_{}_model_{}.npy'
    result = []
    for i in range(1,6):
        for j in range(1,6):
            result.append(path + features_file_names_format.format(i, j))
    return result


def entry_to_fasta(entry, id, path):
    FIELDS = ['Name', 'Wildtype']
    unique_id = '|'.join([entry[field] for field in FIELDS] + [str(id)]).replace(' ','').replace('/','-')

    first_line = '>' + unique_id
    sequence = entry['Sequence']

    fasta_content = first_line + '\n' + sequence

    fasta_path = os.path.join(path, generate_fasta_name(entry, id, path))
    with open(fasta_path, 'w') as f:
        f.write(fasta_content)


def entry_to_pdb_name(entry, id, path):
    return "my name"


def file_to_distance_matrix(input_path, output_path, parser, threas):
    struct = parser.get_structure(input_path, input_path)
    atoms = struct.get_atoms()
    coords = np.array([atom.coord for atom in atoms]).astype(np.float32)
    dists = 1 / distance.cdist(coords, coords)
    dists[dists == np.inf] = 0
    # dists[dists < threas] = -1
    np.save(output_path, dists)


def generate_distance_matrices_from_folder(input_path, output_path):
    parser = PDB.PDBParser()
    for (dirpath, dirnames, filenames) in os.walk(input_path):
        for filename in filenames:
            pdb_path = os.path.join(dirpath, filename)
            npy_path = os.path.join(output_path, (filename).replace('.pdb', '_dists.npy'))
            file_to_distance_matrix(pdb_path, npy_path, parser, 16)
            # print(pdb_path, npz_path)


def generate_fasta_files(df, path):
    for i in range(len(df)):
        entry_to_fasta(df.iloc[i], i, path)


def main():
    generate_distance_matrices_from_folder("/cs/labs/dina/meitar/rhodopsins/pdbs/",
                                           "/cs/labs/dina/meitar/rhodopsins/graphs/")
    # df = pd.read_excel(EXCEL_PATH)
    # generate_fasta_files(df, FASTAS_PATH)

if __name__ == "__main__":
    main()
