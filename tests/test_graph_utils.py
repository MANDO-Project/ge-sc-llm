import pickle

import networkx as nx

from sco_models.graph_utils import load_hetero_nx_graph


def test_load_hetero_nx_graph_reads_pickle_graph(tmp_path):
    graph = nx.MultiDiGraph()
    graph.add_node("n0", node_type="contract")
    graph.add_node("n1", node_type="function")
    graph.add_edge("n0", "n1", edge_type="contains")
    graph_path = tmp_path / "graph.gpickle"
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)

    loaded = load_hetero_nx_graph(graph_path)

    assert list(loaded.nodes) == [0, 1]
    assert loaded.nodes[0]["node_hetero_id"] == 0
    assert loaded.nodes[1]["node_hetero_id"] == 0
