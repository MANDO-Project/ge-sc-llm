# Classifier Quickstart

This guide is for the refactored direct classifier scripts:

- `python graph_classifier.py ...` for contract-level graph detection.
- `python node_classifier.py ...` for node, line, or function-level detection.


## 1. Prepare The Environment

Use Python 3.7 or later with the CUDA/PyTorch/DGL versions listed in the README.

```bash
pip install -r requirements.txt \
  -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html \
  -f https://data.pyg.org/whl/torch-1.8.0+cu111.html \
  -f https://data.dgl.ai/wheels/repo.html
```

Verify the scripts are visible:

```bash
python graph_classifier.py --help
python node_classifier.py --help
```

## 2. Check The Default Asset Bundle

The default pretrained checkpoints and compressed graphs are expected under `mando-assets/` (link to [download](https://drive.google.com/drive/folders/1Ne3ozncJK1Oqaf6hlF_NdQAiS4VgWulx?usp=drive_link)). These assets use `nodetype` node features, so no external embedding feature file is required.

Expected layout:

```text
mando-assets/
  checkpoints/
    graph_detection/nodetype/
    node_detection/nodetype/
  graphs/
    graph_detection/
    node_detection/
```

Supported vulnerability names:

```text
access_control
arithmetic
denial_of_service
front_running
reentrancy
time_manipulation
unchecked_low_level_calls
```

## 3. Run Graph Classification Inference

### Batch inference
Choose a vulnerability type and set the matching paths. This example uses `reentrancy`.

```bash
BUG=reentrancy

python graph_classifier.py --test \
  --node_feature nodetype \
  --compressed_graph "mando-assets/graphs/graph_detection/${BUG}_tree_sitter_cfg_cg_compressed_graphs.gpickle" \
  --checkpoint "mando-assets/checkpoints/graph_detection/nodetype/${BUG}_tree_sitter_cfg_cg_hgt.pth" \
  --label "PATH/TO/graph_labels.json" \
  --testset "PATH/TO/test/source_files"
```
python graph_classifier.py --test \
  --node_feature nodetype \
  --compressed_graph "mando-assets/graphs/graph_detection/${BUG}_tree_sitter_cfg_cg_compressed_graphs.gpickle" \
  --checkpoint "mando-assets/checkpoints/graph_detection/nodetype/${BUG}_tree_sitter_cfg_cg_hgt.pth" \
  --label "mando-assets/graphs/graph_detection/${BUG}_graph_labels.json" \
  --testset "experiments/ge-sc-data/source_code/${BUG}/curated"



`--label` should point to the graph label JSON for the contracts being evaluated. `--testset` should point to the Solidity files or split directory used to select the evaluation subset.

### Expected output
The expected output is the predicted labels of the test set in JSON format.

```json
{
  "contract1.sol": 0,
  "contract2.sol": 1,
  ...
}
```

Where `0` indicates no vulnerability and `1` indicates a vulnerability.


## 4. Run Node Classification Inference

### Batch inference
For node detection, use the matching node graph and node checkpoint. This example again uses `reentrancy`.

```bash
BUG=reentrancy

python node_classifier.py --test \
  --node_feature nodetype \
  --compressed_graph "mando-assets/graphs/node_detection/${BUG}_tree_sitter_cfg_compressed_graphs.gpickle" \
  --checkpoint "mando-assets/checkpoints/node_detection/nodetype/${BUG}_tree_sitter_cfg_cg_hgt.pth" \
  --testset "PATH/TO/test/source_files"
```

### Expected output
The expected output is the predicted labels of the test set in JSON format.

```json
{
  "contract1.sol": {
    {'nodeID': 'node1', 'label': 0, 'lines': [1, 2, 3]},
    {'nodeID': 'node2', 'label': 1, 'lines': [4, 5, 6]},
    ...
  },
  "contract2.sol": {
    {'nodeID': 'node1', 'label': 0, 'lines': [7, 8, 9]},
    {'nodeID': 'node2', 'label': 0, 'lines': [10, 11, 12]},
    ...
  },
  ...
}
```

and a summary of the test metrics included:

```text
Buggy nodes in test: 6572/11948 (55.005021760964176%)
Testing on 11948 nodes
Test Micro f1:   0.4746 | Test Macro f1:   0.4559 | Test Accuracy:   0.4746
Classification report
               precision    recall  f1-score   support

           0     0.4487    0.7334    0.5568      5376
           1     0.5467    0.2629    0.3551      6572

    accuracy                         0.4746     11948
   macro avg     0.4977    0.4982    0.4559     11948
weighted avg     0.5026    0.4746    0.4458     11948

Confusion matrix
 [[3943 1433]
 [4844 1728]]
```

Where `0` indicates no vulnerability and `1` indicates a vulnerability.


## 5. Single file inference

Use `infer_smart_contract.py` when you want to analyze one Solidity source file instead of an existing test split. This mode generates a tree-sitter CFG for the file, loads the tree-sitter node-detection checkpoint for the selected vulnerability type, and runs `HGTVulNodeClassifier.extend_forward` on the generated graph.

```bash
BUG=access_control
SOURCE=experiments/ge-sc-data/source_code/access_control/cleaned_buggy_curated/arbitrary_location_write_simple.sol

python infer_smart_contract.py "$SOURCE" \
  --bug_type "$BUG" \
  --top_k 10 \
  --output_json docs/contract_inference_access_control_example.json
```

By default, the script resolves assets from:

```text
mando-assets/graphs/node_detection/${BUG}_tree_sitter_cfg_compressed_graphs.gpickle
mando-assets/checkpoints/node_detection/nodetype/${BUG}_tree_sitter_cfg_cg_hgt.pth
```

Override those paths with `--compressed_graph` and `--checkpoint` if you are testing a custom model. The output is node-level: `contract_prediction` is an aggregate that becomes `vulnerable` when at least one generated node is predicted vulnerable. Inspect the ranked `nodes` list for line numbers, node type, source snippet, and vulnerability probability.

Example output is attached at [`docs/contract_inference_access_control_example.json`](contract_inference_access_control_example.json).


## 6. Run Direct Training

Training uses the same direct scripts, but omit `--test` and set `--output_models`.

```bash
BUG=reentrancy

python graph_classifier.py \
  --node_feature nodetype \
  --compressed_graph "mando-assets/graphs/graph_detection/${BUG}_cfg_cg_compressed_graphs.gpickle" \
  --label "PATH/TO/graph_labels.json" \
  --output_models "models/graph_detection/nodetype/${BUG}_hgt.pth" \
  --num_epochs 50 \
  --k_folds 5 \
  --seed 1
```

```bash
BUG=reentrancy

python node_classifier.py \
  --node_feature nodetype \
  --compressed_graph "mando-assets/graphs/node_detection/${BUG}_cfg_cg_compressed_graphs.gpickle" \
  --output_models "models/node_detection/nodetype/${BUG}_hgt.pth" \
  --num_epochs 50 \
  --k_folds 5 \
  --seed 1
```

## 7. Common Checks

- Use `--node_feature nodetype` with `mando-assets` defaults.
