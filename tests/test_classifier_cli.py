from pathlib import Path

import pytest

from sco_models.classifier_cli import (
    GRAPH_RUNTIME_DEFAULTS,
    NODE_RUNTIME_DEFAULTS,
    build_graph_parser,
    build_node_parser,
    finalize_graph_args,
    finalize_node_args,
    validate_paths,
)
from graph_classifier import _graph_test_ids
from node_classifier import _node_test_ids


def test_graph_defaults_use_mando_assets_nodetype_reentrancy():
    parser = build_graph_parser()
    args = finalize_graph_args(parser.parse_args([]))

    assert args["bug_type"] == "reentrancy"
    assert args["node_feature"] == "nodetype"
    assert args["compressed_graph"] == (
        "mando-assets/graphs/graph_detection/reentrancy_cfg_cg_compressed_graphs.gpickle"
    )
    assert args["checkpoint"] == (
        "mando-assets/checkpoints/graph_detection/nodetype/reentrancy_hgt.pth"
    )
    assert args["batch_size"] == GRAPH_RUNTIME_DEFAULTS["batch_size"]
    assert "device" in args


def test_graph_tree_sitter_defaults_keep_matching_asset_family():
    parser = build_graph_parser()
    args = finalize_graph_args(parser.parse_args(["--tree_sitter", "--bug_type", "access_control"]))

    assert args["compressed_graph"] == (
        "mando-assets/graphs/graph_detection/"
        "access_control_tree_sitter_cfg_cg_compressed_graphs.gpickle"
    )
    assert args["checkpoint"] == (
        "mando-assets/checkpoints/graph_detection/nodetype/"
        "access_control_tree_sitter_cfg_cg_hgt.pth"
    )


def test_explicit_graph_paths_are_preserved():
    parser = build_graph_parser()
    args = finalize_graph_args(
        parser.parse_args(
            [
                "--compressed_graph",
                "custom/graph.gpickle",
                "--checkpoint",
                "custom/model.pth",
                "--node_feature",
                "line",
            ]
        )
    )

    assert args["compressed_graph"] == "custom/graph.gpickle"
    assert args["checkpoint"] == "custom/model.pth"
    assert args["node_feature"] == "line"


def test_node_defaults_use_mando_assets_nodetype_reentrancy():
    parser = build_node_parser()
    args = finalize_node_args(parser.parse_args([]))

    assert args["bug_type"] == "reentrancy"
    assert args["node_feature"] == "nodetype"
    assert args["compressed_graph"] == (
        "mando-assets/graphs/node_detection/reentrancy_cfg_cg_compressed_graphs.gpickle"
    )
    assert args["checkpoint"] == (
        "mando-assets/checkpoints/node_detection/nodetype/reentrancy_hgt.pth"
    )
    assert args["batch_size"] == NODE_RUNTIME_DEFAULTS["batch_size"]


def test_node_tree_sitter_defaults_keep_matching_asset_family():
    parser = build_node_parser()
    args = finalize_node_args(parser.parse_args(["--tree_sitter", "--bug_type", "front_running"]))

    assert args["compressed_graph"] == (
        "mando-assets/graphs/node_detection/"
        "front_running_tree_sitter_cfg_compressed_graphs.gpickle"
    )
    assert args["checkpoint"] == (
        "mando-assets/checkpoints/node_detection/nodetype/"
        "front_running_tree_sitter_cfg_cg_hgt.pth"
    )


def test_validate_paths_reports_argument_name(tmp_path):
    missing = tmp_path / "missing.pth"

    with pytest.raises(FileNotFoundError, match=f"--checkpoint not found: {missing}"):
        validate_paths({"checkpoint": str(missing)}, ["checkpoint"])


def test_validate_paths_accepts_existing_file(tmp_path):
    existing = tmp_path / "model.pth"
    existing.write_text("weights", encoding="utf8")

    validate_paths({"checkpoint": str(existing)}, ["checkpoint"])


def test_graph_test_ids_match_label_contract_names(tmp_path):
    testset = tmp_path / "source_files"
    testset.mkdir()
    (testset / "A.sol").write_text("contract A {}", encoding="utf8")
    dataset = type("Dataset", (), {"graphs": ["A.sol", "B.sol"], "__len__": lambda self: 2})()

    assert _graph_test_ids(dataset, str(testset)) == [0]


def test_node_test_ids_falls_back_to_all_nodes_when_testset_missing():
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2])

    assert _node_test_ids(graph, "missing/source_files") == [0, 1, 2]
