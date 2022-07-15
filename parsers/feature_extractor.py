import numpy as np
import pandas as pd

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


def amino_acid_from_node_name(node_name):
    return node_name[-3:]


def nodes_to_vec(neighbors_list):

    result = np.zeros(len(AMINO_ACIDS_DICTIONARY))
    for node in neighbors_list:
        result[AMINO_ACIDS_DICTIONARY.index(node)] += 1

    return result.tolist()


# You provide a collection of functions
# that take in the node name and metadata dictionary,
# and return a pandas Series:
def node_neighbours_to_pd(n, d):
    neighbours_statistics = nodes_to_vec(d['neighbors'])
    return pd.Series({"residue number": neighbours_statistics}, name=n)


feature_generator_funcs = [
    node_neighbours_to_pd,
]
