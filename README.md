# GO-MO: A large-scale graph-augmented traffic dataset for data-driven spatio-temporal traffic analysis

This repository contains utilities and transforms used with the GO-MO traffic dataset.

It provides helpers to:
- Convert a road network graph into a line graph suitable for routing/ML (`to_line_graph`).
- Export adjacency matrices for selected sensors/nodes (`to_adj_matrix`).
- Convert a NetworkX directed graph to PyTorch Geometric `Data` (`to_torch_geometric_data`).
- Print quick connectivity/size stats for graphs (`print_graph_stats`).


## Installation / Setup

This repo uses a simple `src/` layout. Add `src` to your `PYTHONPATH` before
running scripts or importing the transforms.

Examples:

- PowerShell (Windows):
  ```powershell
  $env:PYTHONPATH = ".\src"
  python -c "from transforms import to_line_graph, to_adj_matrix; print('OK')"
  ```

- Unix shells (bash/zsh):
  ```bash
  export PYTHONPATH=./src
  python -c "from transforms import to_line_graph, to_adj_matrix; print('OK')"
  ```

Optional: create and activate a virtual environment and install requirements:

```bash
pip install -r requirements.txt
```


## Generating the largest dataset

The script `src/generate_largest_dataset.py` downloads GO-MO traffic data and
the route-based graph from the Hugging Face dataset repository, filters sensors
with incomplete coverage, normalizes traffic intensity, and writes to `output/`
the artifacts needed to run the validation experiments with the Largest Benchmark.
The Largest Benchmark is part of
[LargeST](https://github.com/liuxu77/LargeST); these generated files were used
for the technical validation experiments of the GO-MO dataset.

Before running it, complete the installation/setup steps above.

The Hugging Face dataset is public, so a token is not required. Setting
`HF_API_TOKEN_GOMO` can still help with Hugging Face rate limits and faster
downloads:

```powershell
$env:HF_API_TOKEN_GOMO = "<your-token>"
```

Run the generator from the repository root:

```powershell
python .\src\generate_largest_dataset.py
```

The main configuration values are currently defined near the top of the script:

- `output_folder`: output directory, defaults to `output`.
- `years`: traffic CSV years to download, defaults to `[2024]`.
- `sensors`: optional sensor subset; `None` uses all available sensors.
- `allow_non_complete_years`: when `False`, drops newborn, dying, and
  temporarily offline sensors.
- `sequence_length`: history and prediction window length, defaults to `12`.
- `splits`: train/validation/test split proportions, currently `0.6/0.2/0.2`.

The script writes:

- `output/his.npz`: normalized traffic tensor plus scaler mean and standard
  deviation.
- `output/idx_train.npy`, `output/idx_val.npy`, `output/idx_test.npy`: temporal
  split indexes.
- `output/metraq_rn_adj.npy`: adjacency matrix built from `routes-graph.pkl`.


## Graph transforms

The graph conversion helpers are exposed from the `transforms` package. With
`src` on `PYTHONPATH` you can:

```python
from transforms import to_line_graph, to_adj_matrix, to_torch_geometric_data
```

- `to_line_graph(G_in, disallow_uturn=True, turn_cost_fn=None)`
  - Input: `G_in` - a directed multigraph (`nx.MultiDiGraph`) of the road network.
  - Output: `(G_line, edge_node, node_edge)` where `G_line` is a `nx.DiGraph` whose nodes correspond to original edges.
  - Notes: set `disallow_uturn=False` to allow U-turn transitions. Provide a custom `turn_cost_fn(G, e_from, e_to)` to override edge attributes such as `size` or `turn_angle`.

- `to_adj_matrix(graph, sensors, self_loops=True)`
  - Builds a NumPy adjacency matrix (weighted) for the given `sensors` subset. If `sensors=None`, uses all nodes.
  - When `self_loops=True`, the identity is added to the diagonal for the selected nodes.

- `to_torch_geometric_data(G)`
  - Converts a directed NetworkX graph (e.g., the line graph) to a `torch_geometric.data.Data` with:
    - `x`: node features `[lanes, length, maxspeed, bearing]`
    - `edge_attr`: edge features `[size, turn_angle]`
    - `pos`: node coordinates `[x, y]`

- `tools.print_graph_stats(G)`
  - Prints basic stats: nodes, edges, average degree, weak/strong connectivity, components, isolates, and undirected connectivity check.


## Citation

If you use the released dataset files, metadata, or graph files, please cite the dataset record:

```bibtex
@dataset{maria_arribas2025gomo,
  author       = {Mar{\'i}a-Arribas, David and Cuesta-Infante, Alfredo and Pantrigo, Juan J.},
  title        = {{GO-MO: A large-scale graph-augmented traffic dataset for data-driven spatio-temporal traffic analysis}},
  year         = {2025},
  publisher    = {Hugging Face},
  version      = {1.0},
  doi          = {10.57967/hf/7201},
  url          = {https://doi.org/10.57967/hf/7201}
}
```

If you also refer to the scientific description, methodology, data curation process, graph construction, or validation analyses, please cite the associated preprint article as well. The article citation will be updated here once the final published version is available.

```bibtex
@misc{maria_arribas2026large_scale,
  author       = {Mar{\'i}a-Arribas, David and Pantrigo, Juan J. and Cuesta-Infante, Alfredo},
  title        = {{A large-scale graph-augmented traffic dataset for data-driven spatio-temporal traffic analysis}},
  year         = {2026},
  note         = {Preprint},
  publisher    = {Research Square},
  doi          = {10.21203/rs.3.rs-8670080/v1},
  url          = {https://doi.org/10.21203/rs.3.rs-8670080/v1}
}
```


## License

This repository is released under the MIT License. 
