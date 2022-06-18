import numpy as np

AMINO_ACIDS_DICTIONARY = [
    'ALA',
    'ARG',
    'ASN',
    'ASP',
    'CYS',
    'GLN',
    'GLU',
    'GLY',
    'HIS',
    'ILE',
    'LEU',
    'LYS',
    'MET',
    'PHE',
    'PRO',
    'SER',
    'THR',
    'TRP',
    'TYR',
    'VAL'
]

def nodes_to_vec(neighbors_list):

    result = np.zeros(len(AMINO_ACIDS_DICTIONARY))
    for node in neighbors_list:
        result[AMINO_ACIDS_DICTIONARY.index(node)] += 1

    return result.tolist()
