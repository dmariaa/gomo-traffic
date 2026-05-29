from typing import Optional

import networkx as nx
import numpy as np


def to_adj_matrix(graph: nx.Graph, sensors: Optional[list], self_loops: bool = True) -> np.ndarray:
    """
    Converts the given GO-MO graph into a numpy adjacency matrix for the specified sensors.
    Self-loops are added to nodes corresponding to the sensors, and the adjacency matrix of the resulting graph is returned.

    The adjacency matrix is constructed based on the nodes specified in the sensors parameter.
    Only the subgraph consisting of these nodes is considered, inclusive of any weights on the edges.
    If sensors is None, the entire graph is considered.
    Self-loops will be added to these nodes.

    :param graph: The input GO-MO graph.
    :type graph: nx.Graph
    :param sensors: A list of nodes representing the subset of the graph for which the adjacency
        matrix is computed. If None, the entire graph is returned.
    :type sensors: list
    :param self_loops: If True, add self-loops (identity) to the adjacency matrix diagonal for the
        selected nodes.
    :type self_loops: bool
    :return: A numpy ndarray representing the weighted adjacency matrix of the largest
        connected subgraph corresponding to the nodes (sensors) provided.
    :rtype: np.ndarray
    """
    sensors = sensors or list(graph.nodes)

    # get adj matrix
    adj_matrix = nx.to_numpy_array(graph, nodelist=sensors, weight='weight')

    # add self loops if requested
    if self_loops:
        adj_matrix += np.eye(len(sensors))

    return adj_matrix



