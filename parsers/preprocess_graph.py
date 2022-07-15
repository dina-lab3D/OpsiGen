import parsers.feature_extractor as feature_extractor


def add_neighbours_feature(graph):
    """
    Add a feature to each node in the graph that will save the amino acids of the neighbors
    :param graph: nx graph
    """
    for node in graph.nodes():
        graph.nodes[node]['neighbors'] = []
        for neighbor_name in graph.neighbors(node):
            graph.nodes[node]['neighbors'].append(feature_extractor.amino_acid_from_node_name(neighbor_name))


def update_graph_features(graph):
    """
    preprocess the graph
    :param graph: nx graph
    """
    add_neighbours_feature(graph)
