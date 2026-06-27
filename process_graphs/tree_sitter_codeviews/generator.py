from pathlib import Path

import networkx as nx

from .codeviews.CFG.CFG_driver import CFGDriver


SUPPORTED_LANGUAGES = {"solidity"}


def _validate_language(language):
    if language not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(f"unsupported language: {language}. Supported languages: {supported}")


def _annotate_cfg_graph(graph, source_name):
    nx.set_node_attributes(graph, source_name, "source_file")
    source_lines = {}
    nodes_to_remove = []

    for node_id, node_data in graph.nodes(data=True):
        if "node_type" not in node_data or "label" not in node_data:
            nodes_to_remove.append(node_id)
            continue
        try:
            first_line = int(node_data["label"].split("-")[-2])
            last_line = int(node_data["label"].split("-")[-1])
        except (ValueError, IndexError):
            nodes_to_remove.append(node_id)
            continue
        source_lines[node_id] = {
            "node_source_code_lines": list(range(first_line, last_line + 1))
        }

    graph.remove_nodes_from(nodes_to_remove)
    nx.set_node_attributes(graph, source_lines)
    return graph


def generate_cfg_from_source(source_code, source_name="contract_0.sol", language="solidity"):
    _validate_language(language)
    if not source_code or not source_code.strip():
        raise ValueError("source_code must not be empty")

    cfg_driver = CFGDriver(language, source_code, output_file=None)
    return _annotate_cfg_graph(cfg_driver.graph, source_name)


def generate_cfg_from_file(source_path, language="solidity"):
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"source file not found: {path}")
    source_code = path.read_text(encoding="utf8")
    return generate_cfg_from_source(source_code, source_name=path.name, language=language)


def generate_cfgs_from_directory(source_dir, pattern="*.sol", limit=None, language="solidity"):
    directory = Path(source_dir)
    if not directory.exists():
        raise FileNotFoundError(f"source directory not found: {directory}")
    paths = sorted(directory.glob(pattern))
    if limit is not None:
        paths = paths[:limit]
    return {path.name: generate_cfg_from_file(path, language=language) for path in paths}
