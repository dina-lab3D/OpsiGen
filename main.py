from proteingraph import read_pdb
import proteingraph.conversion
import Bio.PDB
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import networkx as nx

PATH = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/graphs"
PROTEIN_NAME = "1ATN"

# You provide a collection of functions
# that take in the node name and metadata dictionary,
# and return a pandas Series:
def my_func(n, d):
    print("data is ", str(d))
    return pd.Series({}, name=n)


def my_func2(n, d):
    print("data is ", str(d))
    return pd.Series({}, name=n)


def my_func3(n, d):
    print("data is ", str(d))
    return pd.Series({}, name=n)


def to_arr(graph):
    return xr.DataArray(graph)

funcs = [
    to_arr
]


def main():
    pdbl = Bio.PDB.PDBList()
    native_pdb = pdbl.retrieve_pdb_file(pdb_code=PROTEIN_NAME, pdir=PATH, file_format='pdb')
    graph = read_pdb("{}/{}".format(PATH, "pdb1atn.ent"))
    print(graph)
    # adj_da = proteingraph.conversion.generate_adjacency_tensor(graph, funcs)
    # adj_da.plot()
    nx.draw(graph)
    plt.savefig("mygraph.png")
    print("Hello world")


if __name__ == "__main__":
    main()
