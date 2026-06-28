import argparse
import os
from pathlib import Path

import torch


BUG_TYPES = (
    "access_control",
    "arithmetic",
    "denial_of_service",
    "front_running",
    "reentrancy",
    "time_manipulation",
    "unchecked_low_level_calls",
)

GRAPH_RUNTIME_DEFAULTS = {
    "lr": 0.0005,
    "num_heads": 8,
    "hidden_units": 8,
    "dropout": 0.6,
    "weight_decay": 0.001,
    "batch_size": 256,
    "patience": 10,
}

NODE_RUNTIME_DEFAULTS = {
    "lr": 0.0005,
    "num_heads": 8,
    "hidden_units": 8,
    "dropout": 0.6,
    "weight_decay": 0.001,
    "batch_size": 512,
}


def _device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _asset_path(asset_root, *parts):
    return str(Path(asset_root, *parts))


def graph_asset_paths(asset_root, bug_type, tree_sitter=False):
    graph_name = f"{bug_type}_cfg_cg_compressed_graphs.gpickle"
    checkpoint_name = f"{bug_type}_hgt.pth"
    if tree_sitter:
        graph_name = f"{bug_type}_tree_sitter_cfg_cg_compressed_graphs.gpickle"
        checkpoint_name = f"{bug_type}_tree_sitter_cfg_cg_hgt.pth"
    return {
        "compressed_graph": _asset_path(asset_root, "graphs", "graph_detection", graph_name),
        "checkpoint": _asset_path(
            asset_root,
            "checkpoints",
            "graph_detection",
            "nodetype",
            checkpoint_name,
        ),
    }


def node_asset_paths(asset_root, bug_type, tree_sitter=False):
    graph_name = f"{bug_type}_cfg_cg_compressed_graphs.gpickle"
    checkpoint_name = f"{bug_type}_hgt.pth"
    if tree_sitter:
        graph_name = f"{bug_type}_tree_sitter_cfg_compressed_graphs.gpickle"
        checkpoint_name = f"{bug_type}_tree_sitter_cfg_cg_hgt.pth"
    return {
        "compressed_graph": _asset_path(asset_root, "graphs", "node_detection", graph_name),
        "checkpoint": _asset_path(
            asset_root,
            "checkpoints",
            "node_detection",
            "nodetype",
            checkpoint_name,
        ),
    }


def _add_common_arguments(parser, default_log_dir, default_output_models):
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--asset_root",
        type=str,
        default="mando-assets",
        help="Root directory for pretrained checkpoints and compressed graphs",
    )
    parser.add_argument(
        "--bug_type",
        type=str,
        choices=BUG_TYPES,
        default="reentrancy",
        help="Vulnerability type used to resolve default mando-assets paths",
    )
    parser.add_argument(
        "--tree_sitter",
        action="store_true",
        help="Use tree-sitter generated default graph/checkpoint asset names",
    )

    archive_params = parser.add_argument_group(
        title="Storage",
        description="Directories for util results",
    )
    archive_params.add_argument(
        "-ld",
        "--log_dir",
        type=str,
        default=default_log_dir,
        help="Directory for saving training logs and visualization",
    )
    archive_params.add_argument(
        "--output_models",
        type=str,
        default=default_output_models,
        help="Where you want to save your models",
    )

    train_option_params = parser.add_argument_group(
        title="Optional configures",
        description="Advanced options",
    )
    train_option_params.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Config number of epochs",
    )
    train_option_params.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Config for cross validate strategy",
    )
    train_option_params.add_argument(
        "--test",
        action="store_true",
        help="Set true if you only want to run test phase",
    )
    train_option_params.add_argument(
        "--non_visualize",
        action="store_true",
        help="Whether you want to visualize the metrics",
    )


def build_graph_parser():
    parser = argparse.ArgumentParser("MANDO Graph Classifier")
    _add_common_arguments(
        parser,
        default_log_dir="./logs/graph_classification",
        default_output_models="./models/graph_detection/nodetype/reentrancy_hgt.pth",
    )

    dataset_params = parser.add_argument_group(title="Dataset", description="Dataset paths")
    dataset_params.add_argument(
        "--compressed_graph",
        type=str,
        default=None,
        help="Compressed graphs of dataset which was extracted by graph helper tools",
    )
    dataset_params.add_argument(
        "--dataset",
        type=str,
        default="./dataset/aggregate/source_code",
        help="Directory of all source code files used to extract the compressed graph",
    )
    dataset_params.add_argument(
        "--testset",
        type=str,
        default="./dataset/smartbugs/source_code",
        help="Directory of source code files used for testing",
    )
    dataset_params.add_argument(
        "--label",
        type=str,
        default="./dataset/aggregate/labels.json",
        help="Label of sources in source code storage",
    )
    dataset_params.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint of trained models",
    )

    node_feature_params = parser.add_argument_group(
        title="Node feature",
        description="Define the way to get node features",
    )
    node_feature_params.add_argument(
        "--feature_compressed_graph",
        type=str,
        default=None,
        help="Compressed graph used by a HAN feature extractor",
    )
    node_feature_params.add_argument(
        "--feature_extractor",
        type=str,
        default=None,
        help='Feature extractor path for "gae", "line", "node2vec", or HAN features',
    )
    node_feature_params.add_argument(
        "--node_feature",
        type=str,
        default="nodetype",
        help='One of "nodetype", "metapath2vec", "han", "gae", "line", "node2vec"',
    )
    return parser


def build_node_parser():
    parser = argparse.ArgumentParser("MANDO Node Classifier")
    _add_common_arguments(
        parser,
        default_log_dir="./logs/node_classification",
        default_output_models="./models/node_detection/nodetype/reentrancy_hgt.pth",
    )

    dataset_params = parser.add_argument_group(title="Dataset", description="Dataset paths")
    dataset_params.add_argument(
        "--compressed_graph",
        type=str,
        default=None,
        help="Compressed graphs of dataset which was extracted by graph helper tools",
    )
    dataset_params.add_argument(
        "--dataset",
        type=str,
        default="./dataset/aggregate/source_code",
        help="Directory of all source code files used to extract the compressed graph",
    )
    dataset_params.add_argument(
        "--testset",
        type=str,
        default="./dataset/smartbugs/source_code",
        help="Directory of source code files used for testing",
    )
    dataset_params.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint of trained models",
    )

    node_feature_params = parser.add_argument_group(
        title="Node feature",
        description="Define the way to get node features",
    )
    node_feature_params.add_argument(
        "--feature_compressed_graph",
        type=str,
        default="./dataset/aggregate/compressed_graph/compressed_graphs.gpickle",
        help="Compressed graph used by the first HAN layer when node_feature is han",
    )
    node_feature_params.add_argument(
        "--cfg_feature_extractor",
        type=str,
        default="./models/metapath2vec_cfg/han_fold_1.pth",
        help="Checkpoint of the first HAN layer when node_feature is han",
    )
    node_feature_params.add_argument(
        "--feature_extractor",
        type=str,
        default=None,
        help='Feature extractor path for "gae", "line", "node2vec", or HAN features',
    )
    node_feature_params.add_argument(
        "--node_feature",
        type=str,
        default="nodetype",
        help='One of "nodetype", "metapath2vec", "han", "gae", "line", "node2vec"',
    )
    parser.add_argument("--patience", type=int, default=7, help="Patience for early stopping")
    return parser


def _finalize_args(namespace, runtime_defaults, asset_paths):
    args = vars(namespace)
    args.update(runtime_defaults)
    args["device"] = _device()
    if not args.get("compressed_graph"):
        args["compressed_graph"] = asset_paths["compressed_graph"]
    if not args.get("checkpoint"):
        args["checkpoint"] = asset_paths["checkpoint"]
    return args


def finalize_graph_args(namespace):
    assets = graph_asset_paths(
        namespace.asset_root,
        namespace.bug_type,
        tree_sitter=namespace.tree_sitter,
    )
    return _finalize_args(namespace, GRAPH_RUNTIME_DEFAULTS.copy(), assets)


def finalize_node_args(namespace):
    assets = node_asset_paths(
        namespace.asset_root,
        namespace.bug_type,
        tree_sitter=namespace.tree_sitter,
    )
    return _finalize_args(namespace, NODE_RUNTIME_DEFAULTS.copy(), assets)


def validate_paths(args, required_names):
    for name in required_names:
        value = args.get(name)
        if value and not os.path.exists(value):
            raise FileNotFoundError(f"--{name} not found: {value}")


def ensure_output_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
