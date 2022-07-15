from proteingraph import read_pdb
import proteingraph.conversion
import pandas as pd
import networkx as nx
import models
import utils__feature_extractor
import numpy as np
import torch
import torch_geometric.utils.convert as converter
from torch_geometric.data import Data
import consts

PATH = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/graphs"
PROTEIN_NAME = "1ATN"

# You provide a collection of functions
# that take in the node name and metadata dictionary,
# and return a pandas Series:
def node_neighbours_to_pd(n, d):
    neighbours_statistics = utils__feature_extractor.nodes_to_vec(d['neighbors'])
    return pd.Series({"residue number": neighbours_statistics}, name=n)

funcs = [
    node_neighbours_to_pd,
    # my_func2,
    # my_func3
]

def update_graph_features(graph):

    for node in graph.nodes():
        graph.nodes[node]['neighbors'] = []
        for neighbor in graph.neighbors(node):
            graph.nodes[node]['neighbors'].append(neighbor[-3:])


def graph_to_data(graph):
    update_graph_features(graph)
    F = proteingraph.conversion.generate_feature_dataframe(graph, funcs=funcs, return_array=True)
    edge_data = torch.tensor(np.vstack([np.array(l[0]) for l in F]))
    adj = nx.to_scipy_sparse_matrix(graph)
    print(adj)
    edge_index, _ = converter.from_scipy_sparse_matrix(adj)

    print(edge_data.shape, edge_index.shape)

    return edge_data, edge_index

    # return Data(edge_data, edge_index)


def main():
    print("Hello")


if __name__ == "__main__":
    main()
