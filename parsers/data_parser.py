import proteingraph.conversion
import networkx as nx
import parsers.feature_extractor as feature_extractor
import numpy as np
import torch
import torch_geometric.utils.convert as converter
import parsers.preprocess_graph as preprocess_graph

PATH = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/graphs"
PROTEIN_NAME = "1ATN"


def convert_to_pytorch_geometric(featured_graph, nx_graph):
    """
    return edge_data, edge_index as needed in pytorch geometric
    :param featured_graph: the graph as a matrix
    :param nx_graph: the graph as nx graph
    :return: edge_data, edge_index of the graph
    """
    edge_data = torch.tensor(np.vstack([np.array(l[0]) for l in featured_graph]))
    adj = nx.to_scipy_sparse_matrix(nx_graph)
    edge_index, _ = converter.from_scipy_sparse_matrix(adj)

    return edge_data, edge_index


def graph_to_data(graph):
    """
    from a graph get pytorch geometric data of the graph
    :param graph: the graph to work on
    :return: edge_data, edge_index of the graph
    """
    preprocess_graph.update_graph_features(graph)
    featured_graph = proteingraph.conversion.generate_feature_dataframe(graph, funcs=feature_extractor.feature_generator_funcs,
                                                           return_array=True)

    return convert_to_pytorch_geometric(featured_graph, graph)


def main():
    print("Hello")


if __name__ == "__main__":
    main()
