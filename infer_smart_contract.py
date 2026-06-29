import argparse
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

import networkx as nx

from sco_models.classifier_cli import BUG_TYPES, node_asset_paths, validate_paths


CLASS_NAMES = ("valid", "vulnerable")


def build_parser():
    parser = argparse.ArgumentParser(
        "MANDO single-contract node inference",
        description=(
            "Generate a tree-sitter CFG for one Solidity file and run node-level "
            "inference with HGTVulNodeClassifier.extend_forward."
        ),
    )
    parser.add_argument("source_file", help="Solidity source file to analyze")
    parser.add_argument(
        "--asset_root",
        default="mando-assets",
        help="Root directory for pretrained checkpoints and compressed graphs",
    )
    parser.add_argument(
        "--bug_type",
        choices=BUG_TYPES,
        default="reentrancy",
        help="Vulnerability type used to resolve default tree-sitter node assets",
    )
    parser.add_argument(
        "--compressed_graph",
        default=None,
        help="Training compressed graph used to initialize the checkpoint schema",
    )
    parser.add_argument("--checkpoint", default=None, help="Trained node classifier checkpoint")
    parser.add_argument(
        "--device",
        default=None,
        help='Torch device. Defaults to "cuda:0" when CUDA is available, otherwise "cpu".',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of highest vulnerability-probability nodes to print",
    )
    parser.add_argument("--output_json", default=None, help="Optional path for JSON results")
    parser.add_argument(
        "--verbose_generator",
        action="store_true",
        help="Show tree-sitter graph-generator stdout",
    )
    return parser


def add_inference_defaults(graph):
    graph = graph.copy()
    for _, node_data in graph.nodes(data=True):
        node_data.setdefault("node_info_vulnerabilities", None)
    return graph


def normalize_graph_for_model(graph, model):
    graph = add_inference_defaults(graph)
    supported_node_types = set(model.ntypes_dict)
    supported_edge_types = {etype[1] for etype in model.etypes_dict}
    supported_canonical_etypes = set(model.etypes_dict)

    removed_node_types = {}
    nodes_to_remove = []
    for node_id, node_data in graph.nodes(data=True):
        node_type = node_data.get("node_type")
        if node_type not in supported_node_types:
            nodes_to_remove.append(node_id)
            removed_node_types[node_type] = removed_node_types.get(node_type, 0) + 1
    graph.remove_nodes_from(nodes_to_remove)

    removed_edge_types = {}
    edges_to_remove = []
    for source, target, key, edge_data in graph.edges(keys=True, data=True):
        edge_type = edge_data.get("edge_type")
        if edge_type == "next" and "next_line" in supported_edge_types:
            edge_type = "next_line"
            edge_data["edge_type"] = edge_type
            edge_data["label"] = edge_type
            edge_data["controlflow_type"] = edge_type

        source_type = graph.nodes[source]["node_type"]
        target_type = graph.nodes[target]["node_type"]
        canonical_etype = (source_type, edge_type, target_type)
        reverse_etype = (target_type, edge_type, source_type)
        if (
            edge_type not in supported_edge_types
            or canonical_etype not in supported_canonical_etypes
            or reverse_etype not in supported_canonical_etypes
        ):
            edges_to_remove.append((source, target, key))
            removed_edge_types[str(canonical_etype)] = removed_edge_types.get(str(canonical_etype), 0) + 1
    graph.remove_edges_from(edges_to_remove)

    if graph.number_of_nodes() == 0:
        raise ValueError("No generated nodes match the checkpoint graph schema")

    graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_node_id")
    return graph, {
        "removed_nodes": len(nodes_to_remove),
        "removed_node_types": removed_node_types,
        "removed_edges": len(edges_to_remove),
        "removed_edge_types": removed_edge_types,
    }


def summarize_predictions(source_file, graph, logits, schema_report, args):
    import torch

    probabilities = torch.softmax(logits, dim=1).detach().cpu()
    predictions = torch.argmax(probabilities, dim=1).detach().cpu()
    nodes = []
    for node_id, node_data in graph.nodes(data=True):
        vulnerability_probability = float(probabilities[node_id][1])
        predicted_class = int(predictions[node_id])
        nodes.append(
            {
                "node_id": int(node_id),
                "original_node_id": node_data.get("original_node_id"),
                "node_type": node_data.get("node_type"),
                "method": node_data.get("method"),
                "source_lines": node_data.get("node_source_code_lines", []),
                "code": node_data.get("code"),
                "prediction": CLASS_NAMES[predicted_class],
                "valid_probability": float(probabilities[node_id][0]),
                "vulnerability_probability": vulnerability_probability,
            }
        )

    nodes.sort(key=lambda row: row["vulnerability_probability"], reverse=True)
    vulnerable_nodes = [node for node in nodes if node["prediction"] == "vulnerable"]
    max_probability = nodes[0]["vulnerability_probability"] if nodes else 0.0
    return {
        "source_file": str(source_file),
        "bug_type": args.bug_type,
        "compressed_graph": args.compressed_graph,
        "checkpoint": args.checkpoint,
        "contract_prediction": "vulnerable" if vulnerable_nodes else "valid",
        "max_vulnerability_probability": max_probability,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "vulnerable_node_count": len(vulnerable_nodes),
        "schema_report": schema_report,
        "nodes": nodes,
    }


def print_summary(result, top_k):
    print("Source:", result["source_file"])
    print("Bug type:", result["bug_type"])
    print("Contract prediction:", result["contract_prediction"])
    print("Max vulnerability probability: {:.4f}".format(result["max_vulnerability_probability"]))
    print(
        "Predicted vulnerable nodes: {}/{}".format(
            result["vulnerable_node_count"],
            result["node_count"],
        )
    )
    if result["schema_report"]["removed_nodes"] or result["schema_report"]["removed_edges"]:
        print("Schema normalization:", result["schema_report"])
    if top_k <= 0:
        return
    print("\nTop nodes:")
    for node in result["nodes"][:top_k]:
        lines = node["source_lines"]
        line_label = "{}-{}".format(min(lines), max(lines)) if lines else "unknown"
        code = (node.get("code") or "").strip().replace("\n", " ")
        if len(code) > 100:
            code = code[:97] + "..."
        print(
            "- p={:.4f} class={} type={} lines={} method={} code={}".format(
                node["vulnerability_probability"],
                node["prediction"],
                node["node_type"],
                line_label,
                node.get("method") or "",
                code,
            )
        )


def configure_runtime_cache():
    cache_root = Path(tempfile.gettempdir()) / "mando-inference-cache"
    matplotlib_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def generate_tree_sitter_graph(source_file, verbose=False):
    from process_graphs.tree_sitter_codeviews.generator import generate_cfg_from_file

    if verbose:
        return generate_cfg_from_file(source_file)
    with contextlib.redirect_stdout(io.StringIO()):
        return generate_cfg_from_file(source_file)


def main():
    parser = build_parser()
    args = parser.parse_args()

    assets = node_asset_paths(args.asset_root, args.bug_type, tree_sitter=True)
    if args.compressed_graph is None:
        args.compressed_graph = assets["compressed_graph"]
    if args.checkpoint is None:
        args.checkpoint = assets["checkpoint"]

    source_file = Path(args.source_file)
    validate_paths(
        {
            "source_file": str(source_file),
            "compressed_graph": args.compressed_graph,
            "checkpoint": args.checkpoint,
        },
        ["source_file", "compressed_graph", "checkpoint"],
    )

    configure_runtime_cache()

    import torch
    from sco_models.model_hgt import HGTVulNodeClassifier

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    generated_graph = generate_tree_sitter_graph(
        source_file,
        verbose=args.verbose_generator,
    )
    model = HGTVulNodeClassifier(
        args.compressed_graph,
        feature_extractor=None,
        node_feature="nodetype",
        device=device,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    inference_graph, schema_report = normalize_graph_for_model(generated_graph, model)
    with torch.no_grad():
        logits, _ = model.extend_forward(inference_graph)

    result = summarize_predictions(source_file, inference_graph, logits, schema_report, args)
    print_summary(result, args.top_k)

    if args.output_json:
        output_path = Path(args.output_json)
        if output_path.parent:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf8")
        print("\nWrote JSON:", output_path)


if __name__ == "__main__":
    main()
