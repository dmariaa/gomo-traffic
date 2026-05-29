import networkx as nx


def print_graph_stats(G: nx.Graph):    
    """Print basic statistics for a NetworkX graph.

    This utility prints a small set of connectivity and size metrics for the
    provided graph. For empty graphs, all metrics are reported as zero/False
    and the function returns early.

    Metrics printed
    - Number of nodes and edges
    - Average degree (computed as m / n)
    - Connectivity flags for directed graphs: weakly/strongly connected
    - Undirected connectivity of ``G`` viewed as an undirected graph
    - Number of weakly/strongly connected components
    - Number of isolated nodes

    Parameters
    - G (nx.Graph): The input graph. For weak/strong connectivity and their
      component counts, ``G`` is expected to be a directed graph
      (``nx.DiGraph``). On undirected graphs, those directed metrics may not be
      meaningful or supported by NetworkX. Undirected connectivity is computed
      on an undirected view of ``G`` with a safety guard.

    Returns
    - None. Results are printed to stdout.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    print(f"Number of nodes: {n}")
    print(f"Number of edges: {m}")

    if n == 0:
        print("Average degree: 0.0")
        print("Connectivity (weak or strong): False")
        print("Is connected as undirected graph: False")
        print("Number of weakly connected components: 0")
        print("Number of strongly connected components: 0")
        print("Number of isolated nodes: 0")
        return

    # Average degree via m/n
    avg_deg = m / n
    print(f"Average degree: {avg_deg}")

    # Connectivity checks for directed graphs
    is_weak = nx.is_weakly_connected(G)
    is_strong = nx.is_strongly_connected(G)
    print(f"Connectivity (weak or strong): {is_weak or is_strong}")

    # Undirected connectivity can raise on empty graphs; guard with try/except just in case
    try:
        und_connected = nx.is_connected(G.to_undirected(as_view=True))
        print(f"Is connected as undirected graph: {und_connected}")

    except Exception:
        und_connected = False

    print(f"Number of weakly connected components: {nx.number_weakly_connected_components(G)}")
    print(f"Number of strongly connected components: {nx.number_strongly_connected_components(G)}")
    print(f"Number of isolated nodes: {len(list(nx.isolates(G)))}")


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
