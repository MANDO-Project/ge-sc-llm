# Classifier Maintainability Refactor Design

## Goal

Refactor the direct classifier entry points, `graph_classifier.py` and `node_classifier.py`, so other users can run training and reproduce inference more easily. Keep the paper-oriented experiment runners, such as `python -m experiments.graph_classification ...`, separate and behaviorally unchanged.

## Scope

In scope:

- Preserve `python graph_classifier.py ...` and `python node_classifier.py ...` as public commands.
- Extract duplicated and crowded code from both scripts into reusable modules under `sco_models/`.
- Make graph and node inference use command-line paths instead of hard-coded local machine paths.
- Keep existing defaults and argument names unless an additional alias improves clarity without breaking callers.
- Improve errors for missing checkpoints, labels, compressed graphs, and feature files.

Out of scope:

- Redesigning model architectures.
- Changing empirical experiment loops under `experiments/`.
- Changing datasets, labels, checkpoint formats, or trained model outputs.
- Adding new dependency management beyond what is needed for tests or smoke checks.

## Architecture

The root scripts become thin wrappers:

- Build an argument parser.
- Merge default runtime configuration.
- Seed PyTorch.
- Dispatch to training or inference.

Shared functionality moves into focused modules:

- `sco_models/classifier_cli.py`: parser helpers and shared runtime defaults.
- `sco_models/classifier_training.py`: fold result containers, optimizer/scheduler setup, checkpoint path handling, and metric recording helpers.
- `sco_models/graph_classifier_runner.py`: graph-specific model construction, training, validation, and inference/evaluation.
- `sco_models/node_classifier_runner.py`: node-specific model construction, masks, source-file selection, training, validation, and inference/evaluation.

Use these filenames for the first pass unless an import-cycle or missing-dependency issue makes one module impossible to load. Keep graph and node logic separated.

## Command Behavior

Training remains the default mode when `--test` is absent. Existing options such as `--compressed_graph`, `--label`, `--feature_extractor`, `--node_feature`, `--output_models`, `--num_epochs`, and `--k_folds` continue to work.

When `--test` is present, the scripts should:

- Validate required input paths.
- Build the model from `--compressed_graph`, `--node_feature`, and feature arguments.
- Load graph classification weights from `--checkpoint`.
- Add `--checkpoint` to `node_classifier.py` as a backward-compatible alias for the current node inference weight path. Keep existing node arguments working.
- Evaluate the requested test split when `--testset` can be mapped to graph or node IDs.
- Print metrics and reports consistently.

## Error Handling

Before expensive model setup, check for missing files and directories that are required for the selected mode. Error messages should name the argument and path, for example: `--checkpoint not found: ./models/foo.pth`.

Avoid silently using absolute developer-machine paths. If a required value is missing, fail early with an actionable message.

## Testing And Verification

Use test-first implementation for behavior-preserving helpers where feasible:

- Parser/default tests for graph and node scripts.
- Unit tests for checkpoint path generation and required-path validation.
- Lightweight tests for source-file to graph/node ID selection using tiny fake data structures.

Manual verification:

- `python graph_classifier.py --help`
- `python node_classifier.py --help`
- Python syntax/import checks for touched modules
- Data-backed smoke runs only if the required graph files, labels, model classes, and checkpoints are available in the checkout.

## Compatibility Notes

Existing experiment commands remain untouched. If shared helpers become useful for experiments later, that should be a separate pass so paper-result orchestration does not change accidentally.
