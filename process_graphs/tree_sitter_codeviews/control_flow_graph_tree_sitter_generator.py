from .generator import generate_cfg_from_source


def tree_sitter_generate_cfg(
    source_code,
    ori_name="contract_0.sol",
    src_language="solidity",
    CFG_output=None,
):
    graph = generate_cfg_from_source(
        source_code,
        source_name=ori_name,
        language=src_language,
    )
    if CFG_output is not None:
        from .utils import postprocessor

        postprocessor.write_to_dot(graph, CFG_output)
    return graph
