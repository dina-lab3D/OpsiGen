from proteingraph import read_pdb
import proteingraph.conversion
# import Bio.PDB
import pandas as pd
# import xarray as xr
# import matplotlib.pyplot as plt
import networkx as nx
# import models
import feature_extractor
import numpy as np


PATH = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/graphs"
PROTEIN_NAME = "1ATN"

# You provide a collection of functions
# that take in the node name and metadata dictionary,
# and return a pandas Series:
def my_func(n, d):
    neighbours_statistics = feature_extractor.nodes_to_vec(d['neighbors'])
    return pd.Series({"residue number": neighbours_statistics}, name=n)


# def my_func2(n, d):
#     print("data is ", str(d))
#     return pd.Series({}, name=n)


# def my_func3(n, d):
#     print("data is ", str(d))
#     return pd.Series({}, name=n)


# def to_arr(graph):
#     return xr.DataArray(graph)

funcs = [
    my_func,
    # my_func2,
    # my_func3
]

def update_graph_features(graph):

    for node in graph.nodes():
        graph.nodes[node]['neighbors'] = []
        for neighbor in graph.neighbors(node):
            graph.nodes[node]['neighbors'].append(neighbor[-3:])

def main():
    # pdbl = Bio.PDB.PDBList()
    # native_pdb = pdbl.retrieve_pdb_file(pdb_code=PROTEIN_NAME, pdir=PATH, file_format='pdb')
    graph = read_pdb("{}/{}".format(PATH, "pdb1atn.ent"))
    update_graph_features(graph)
    print(graph)
    F = proteingraph.conversion.generate_feature_dataframe(graph, funcs=funcs, return_array=True)
    print("The shape is", np.vstack([np.array(l[0]) for l in F]).shape)
    adj = nx.adjacency_matrix(graph).toarray()
    print(adj.shape)
    # adj_da = proteingraph.conversion.generate_adjacency_tensor(graph, funcs)
    # adj_da.plot()
    # nx.draw(graph)
    # data = from_networkx(graph)
    # print(data)
    # conv_model = models.ConvModel(5, 5, 5)
    # conv_model.forward(data)
    # plt.savefig("mygraph.png")


if __name__ == "__main__":
    main()
