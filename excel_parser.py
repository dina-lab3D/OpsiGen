import pandas as pd
import os
import numpy as np
from BIO import PDB
from scipy.spatial import distance

EXCEL_PATH = "/cs/labs/dina/meitar/rhodopsins/excel/data.xlsx"

FASTAS_PATH = "/cs/labs/dina/meitar/rhodopsins/fastas/"


def generate_fasta_name(entry, id, path):
    return str(id) + '-' + entry['Name'].replace('/','.').replace(' ', '_') + '.fasta'


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


def entry_to_distance_matrix(entry, id, path, parser, threas):
    name = entry_to_pdb_name(entry, id, path)
    struct = parser.get_structure(id, name + '.pdb')
    atoms = struct.get_atoms()
    coords = np.array([atom.coord for atom in atoms]).astype(np.int16)
    dists = distance.cdist(coords, coords)
    dists[dists < threas] = -1
    np.savez(name + '.npz', dists)


def generate_distance_matrices(df, path):
    parser = PDB.PDBParser()
    for i in range(len(df)):
        entry_to_distance_matrix(df.iloc[i], i, path, parser, 16)


def generate_fasta_files(df, path):
    for i in range(len(df)):
        entry_to_fasta(df.iloc[i], i, path)

def main():
    df = pd.read_excel(EXCEL_PATH)
    # generate_fasta_files(df, FASTAS_PATH)

if __name__ == "__main__":
    main()
