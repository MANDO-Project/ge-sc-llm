# Tree-Sitter Graph Generator Refactor Design

## Goal

Refactor the tree-sitter-based graph generator under `process_graphs/` so new users can generate Solidity CFG graphs locally and so the classifier refactor can rely on a stable graph-generation API.

## Scope

In scope:

- Keep existing deep imports and `tree_sitter_generate_cfg(...)` working as a compatibility path.
- Add a clearer public API for Solidity CFG generation from source strings, files, and directories.
- Resolve tree-sitter `vendor/` and `build/my-languages.so` paths relative to the installed package instead of old `app/sco/...` container paths.
- Add tests that use real Solidity files from `experiments/ge-sc-data/source_code/access_control/source_files/`.
- Validate generated graphs have the attributes expected by downstream classifiers: `source_file`, `node_type`, `label`, and `node_source_code_lines`.
- Add clear errors for missing files, empty source input, unsupported languages, and missing tree-sitter vendor assets.

Out of scope:

- Rewriting the CFG construction algorithm.
- Refactoring AST, DFG, or combined graph generation beyond import/path fixes needed for package health.
- Changing graph schemas or checkpoint compatibility.
- Adding Slither-based graph generation.

## Public API

Add a focused module, `process_graphs/tree_sitter_codeviews/generator.py`, with:

- `generate_cfg_from_source(source_code: str, source_name: str = "contract_0.sol", language: str = "solidity") -> nx.MultiDiGraph`
- `generate_cfg_from_file(source_path: str | Path, language: str = "solidity") -> nx.MultiDiGraph`
- `generate_cfgs_from_directory(source_dir: str | Path, pattern: str = "*.sol", limit: int | None = None, language: str = "solidity") -> dict[str, nx.MultiDiGraph]`

`tree_sitter_generate_cfg(...)` delegates to `generate_cfg_from_source(...)` to preserve existing callers.

## Path Resolution

`process_graphs/tree_sitter_codeviews/tree_parser/custom_parser.py` derives paths from `Path(__file__).resolve()`:

- package root: `process_graphs/tree_sitter_codeviews/`
- default build directory: `<package root>/build`
- default language library: `<package root>/build/my-languages.so`
- default vendor root: `<package root>/vendor`

Environment variables override defaults when set:

- `TREE_SITTER_LIB_DIR`
- `TREE_SITTER_VENDOR_DIR`

If `tree-sitter-solidity/src/parser.c` is absent, raise `FileNotFoundError` with the checked path.

## Tests

Create `tests/process_graphs/test_tree_sitter_generator.py`.

Tests use this small real fixture:

`experiments/ge-sc-data/source_code/access_control/source_files/0x0a123a23ade571c7a716c596fb8ac4e6bc642e05.sol`

Required coverage:

- `generate_cfg_from_file` returns a non-empty `networkx.MultiDiGraph`.
- Every generated node has `source_file` equal to the fixture filename.
- At least one generated node has `node_source_code_lines` as a non-empty list of positive integers.
- Graph nodes with retained CFG metadata include `node_type` and `label`.
- `generate_cfgs_from_directory(..., limit=2)` returns exactly two graphs with deterministic sorted filenames.
- Missing source file raises `FileNotFoundError`.
- Empty source string raises `ValueError`.
- Unsupported language raises `ValueError`.

## Verification

Run:

- `pytest tests/process_graphs/test_tree_sitter_generator.py -v`
- `python -m pytest tests/process_graphs/test_tree_sitter_generator.py -v`

If the local tree-sitter binary must be rebuilt, the tests allow compilation of `my-languages.so` into `process_graphs/tree_sitter_codeviews/build/`. The tests never require old `app/sco/...` paths.
