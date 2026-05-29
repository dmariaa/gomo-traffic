import networkx as nx
import torch
from torch_geometric.data import Data


def _get_float_value(v):
    if isinstance(v, list):
        return max([float(i) for i in v])
    else:
        return float(v)


def to_torch_geometric_data(G: nx.DiGraph) -> Data:
    node_ids = sorted(G.nodes())
    n = len(node_ids)
    m = G.number_of_edges()

    mapping = dict(zip(node_ids, range(n)))

    nodes_attrs = torch.empty((n, 4), dtype=torch.float)
    pos = torch.zeros((n, 2), dtype=torch.float)
    for id in node_ids:
        node_data = G.nodes[id]
        lanes = _get_float_value(node_data.get('lanes', 1.0))
        length = _get_float_value(node_data.get('length', 0.0))
        speed = _get_float_value(node_data.get('maxspeed', 30.0))
        bearing = _get_float_value(node_data.get('bearing', 0.0))
        nodes_attrs[mapping[id]] = torch.tensor([lanes, length, speed, bearing], dtype=torch.float)
        pos[mapping[id], 0] = float(node_data.get('x'))
        pos[mapping[id], 1] = float(node_data.get('y'))

    edge_index = torch.empty((2, m), dtype=torch.long)
    edge_feat = torch.empty((m, 2), dtype=torch.float)
    for i, (src, dst, edge_data) in enumerate(G.edges(data=True)):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]
        edge_feat[i, 0] = edge_data.get('size')
        edge_feat[i, 1] = edge_data.get('turn_angle')

    data = Data(x=nodes_attrs, edge_index=edge_index, edge_attr=edge_feat)
    data.pos = pos
    data.mapping = mapping
    return data