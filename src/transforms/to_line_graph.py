import math
import pickle

import networkx
import networkx as nx
from shapely.geometry import LineString, Point

from tools import print_graph_stats


def _remove_isolated_nodes(G: nx.Graph) -> nx.Graph:
    isolated = list(nx.isolates(G))
    print(f"{len(isolated)} isolated edge-nodes")
    if isolated: G.remove_nodes_from(isolated)
    print(f"{len(list(nx.isolates(G)))} isolated edge-nodes")
    return G

def _points_bearing(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    xu, yu = p1
    xv, yv = p2
    bearing = math.atan2(yv - yu, xv - xu) * 180.0 / math.pi
    return bearing % 360.0


def _linestring_bearing(ls: LineString, use_tail: bool = True) -> float:
    if use_tail:
        p1 = ls.coords[-2]
        p2 = ls.coords[-1]
    else:
        p1 = ls.coords[0]
        p2 = ls.coords[1]

    return _points_bearing(p1, p2)


def _edge_midpoint(G, u, v, k, data) -> Point:
    # Try to use edge geometry; fallback to straight segment between node coords
    if 'geometry' in data and isinstance(data['geometry'], LineString):
        geom = data['geometry']
    else:
        # build a simple LineString between endpoints
        xu, yu = G.nodes[u]['x'], G.nodes[u]['y']
        xv, yv = G.nodes[v]['x'], G.nodes[v]['y']
        geom = LineString([(xu, yu), (xv, yv)])
    return geom.interpolate(0.5, normalized=True)

def _edge_bearing(G, u, v, k, data, use_tail: bool = True) -> float:
    # Try to use edge geometry; fallback to straight segment between node coords
    if 'geometry' in data and isinstance(data['geometry'], LineString):
        geom = data['geometry']
        return _linestring_bearing(geom, use_tail=use_tail)
    else:
        # build a simple LineString between endpoints
        xu, yu = G.nodes[u]['x'], G.nodes[u]['y']
        xv, yv = G.nodes[v]['x'], G.nodes[v]['y']
        return _points_bearing((xu, yu), (xv, yv))

def _signed_turn_angle(bearing_from: float, bearing_to: float) -> float:
    angle = (bearing_to - bearing_from + 180) % 360 - 180
    return angle

def _default_turn_attrs(G, e_from_data, e_to_data):
    u, v, k, from_data = e_from_data
    _, w, k2, to_data = e_to_data

    bearing_from = _edge_bearing(G, u, v, k, from_data, use_tail=True)
    bearing_to = _edge_bearing(G, v, w, k2, to_data, use_tail=False)
    turn_angle = _signed_turn_angle(bearing_from, bearing_to)

    length = to_data.get('length', 1.0)
    return {'size': float(length), 'turn_angle': float(turn_angle)}


def to_line_graph(G_in, disallow_uturn=True, turn_cost_fn=None):
    G = networkx.DiGraph()
    edge_node = {}
    node_edge = {}
    if turn_cost_fn is None:
        turn_cost_fn = _default_turn_attrs

    for i, (u, v, k, data) in enumerate(G_in.edges(keys=True, data=True)):
        nid = i  # stable integer id; or use (u,v,k) directly if you pº efer
        edge_node[(u, v, k)] = nid
        node_edge[nid] = (u, v, k)

        # Compute a representative point for visualization/debug
        mid = _edge_midpoint(G_in, u, v, k, data)                       # shapely Point
        bearing = _edge_bearing(G_in, u, v, k, data, use_tail=False)    # original edge direction
        x_attr, y_attr = float(mid.x), float(mid.y)

        # Copy edge attributes to the new node, and add convenience fields
        node_data = dict(data)
        node_data.update(dict(src_u=u, src_v=v, src_key=k, x=x_attr, y=y_attr, bearing=bearing, is_edge_node=True))
        G.add_node(nid, **node_data)

    # 2) Add transitions: (u->v,k) -> (v->w,k2) for all out-edges from v
    for (u, v, k, data_from) in G_in.edges(keys=True, data=True):
        n_from = edge_node[(u, v, k)]

        for _, w, k2, data_to in G_in.out_edges(v, keys=True, data=True):
            if disallow_uturn and w == u:
                continue

            n_to = edge_node[(v, w, k2)]
            attrs = dict(turn_cost_fn(G_in, (u, v, k, data_from), (v, w, k2, data_to)) or {})
            G.add_edge(n_from, n_to, **attrs)

    G = _remove_isolated_nodes(G)
    return G, edge_node, node_edge
