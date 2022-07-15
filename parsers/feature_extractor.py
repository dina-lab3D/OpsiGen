import numpy as np
import pandas as pd

# dictionary of ammino acids
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
    """
    Get the ammino acid from the node's name
    :param node_name: The name of the node
    :return: the ammino acid relevant to this node
    """
    return node_name[-3:]


def nodes_to_vec(neighbors_list):
    """
    Given a list of neighbors, return a numpy vector v s.t. v[i] = neighbours with the i'th amino acid
    :param neighbors_list: list of neighbors
    :return: vector representing the neighbors as a list
    """

    result = np.zeros(len(AMINO_ACIDS_DICTIONARY))
    for node in neighbors_list:
        result[AMINO_ACIDS_DICTIONARY.index(node)] += 1

    return result.tolist()


# You provide a collection of functions
# that take in the node name and metadata dictionary,
# and return a pandas Series:
def node_neighbours_to_pd(node, descriptor):
    """
    get a node and the descriptors of the node and return a pandas dataseries
    :param node: the node to work on
    :param descriptor: the descriptor of that node
    :return: pandas dataseries of that node
    """
    neighbours_statistics = nodes_to_vec(descriptor['neighbors'])
    return pd.Series({"residue number": neighbours_statistics}, name=node)


feature_generator_funcs = [
    node_neighbours_to_pd,
]
