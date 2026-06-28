from pathlib import Path

import networkx as nx
import pytest

from process_graphs.tree_sitter_codeviews.generator import (
    generate_cfg_from_file,
    generate_cfg_from_source,
    generate_cfgs_from_directory,
)
from process_graphs.tree_sitter_codeviews.control_flow_graph_tree_sitter_generator import (
    tree_sitter_generate_cfg,
)
import process_graphs.tree_sitter_codeviews.control_flow_graph_tree_sitter_generator as legacy_generator


FIXTURE_DIR = Path("experiments/ge-sc-data/source_code/access_control/source_files")
FIXTURE_FILE = FIXTURE_DIR / "0x0a123a23ade571c7a716c596fb8ac4e6bc642e05.sol"


def test_generate_cfg_from_file_returns_graph_with_expected_node_attributes():
    graph = generate_cfg_from_file(FIXTURE_FILE)

    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
    assert {data["source_file"] for _, data in graph.nodes(data=True)} == {FIXTURE_FILE.name}
    assert all("node_type" in data for _, data in graph.nodes(data=True))
    assert all("label" in data for _, data in graph.nodes(data=True))
    assert any(
        isinstance(data.get("node_source_code_lines"), list)
        and data["node_source_code_lines"]
        and all(line > 0 for line in data["node_source_code_lines"])
        for _, data in graph.nodes(data=True)
    )


def test_generate_cfg_from_source_rejects_empty_source():
    with pytest.raises(ValueError, match="source_code must not be empty"):
        generate_cfg_from_source("", source_name="empty.sol")


def test_generate_cfg_from_file_rejects_missing_path():
    with pytest.raises(FileNotFoundError, match="source file not found"):
        generate_cfg_from_file(FIXTURE_DIR / "missing.sol")


def test_generate_cfg_from_source_rejects_unsupported_language():
    source = FIXTURE_FILE.read_text(encoding="utf8")

    with pytest.raises(ValueError, match="unsupported language"):
        generate_cfg_from_source(source, source_name=FIXTURE_FILE.name, language="vyper")


def test_generate_cfgs_from_directory_is_sorted_and_limited():
    graphs = generate_cfgs_from_directory(FIXTURE_DIR, limit=2)

    expected_names = [path.name for path in sorted(FIXTURE_DIR.glob("*.sol"))[:2]]
    assert list(graphs) == expected_names
    assert len(graphs) == 2
    assert all(isinstance(graph, nx.MultiDiGraph) for graph in graphs.values())


def test_legacy_tree_sitter_generate_cfg_delegates_to_public_api(monkeypatch):
    calls = {}
    expected_graph = nx.MultiDiGraph()

    def fake_generate_cfg_from_source(source_code, source_name, language):
        calls["source_code"] = source_code
        calls["source_name"] = source_name
        calls["language"] = language
        return expected_graph

    monkeypatch.setattr(
        legacy_generator,
        "generate_cfg_from_source",
        fake_generate_cfg_from_source,
    )

    graph = tree_sitter_generate_cfg("contract C {}", ori_name="C.sol", src_language="solidity")

    assert graph is expected_graph
    assert calls == {
        "source_code": "contract C {}",
        "source_name": "C.sol",
        "language": "solidity",
    }
