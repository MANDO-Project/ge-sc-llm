"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.

Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""
import os

import torch
import networkx as nx
import pickle
import json


from sco_models.model_hgt import HGTVulNodeClassifier
from sco_models.utils import score, get_classification_report, get_confusion_matrix, dump_result


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def get_node_ids(graph, source_files):
    file_ids = []
    for node_ids, node_data in graph.nodes(data=True):
        filename = node_data['source_file'].strip('"')
        if filename in source_files:
            file_ids.append(int(node_ids))

    return file_ids


def main(args):
    epochs = args['num_epochs']
    k_folds = args['k_folds']
    # device = args['device']
    if args['device'] == 'mps':
        device = torch.device("mps")
    else:
        device = args['device']

    # kfold = KFold(n_splits=k_folds, shuffle=True)
    train_results = {}
    val_results = {}
    # Get feature extractor
    print('Getting features')
    if args['node_feature'] == 'han':
        feature_extractor = HGTVulNodeClassifier(args['feature_compressed_graph'], feature_extractor=args['cfg_feature_extractor'], node_feature='gae', device=device)
        feature_extractor.load_state_dict(torch.load(args['feature_extractor']))
        feature_extractor.to(device)
        feature_extractor.eval()
    else:
        feature_extractor = args['feature_extractor']

    # nx_graph = nx.read_gpickle(args['compressed_graph'])
    with open(args['compressed_graph'], 'rb') as f:
        nx_graph = pickle.load(f)
    number_of_nodes = len(nx_graph)
    model = HGTVulNodeClassifier(args['compressed_graph'], feature_extractor=feature_extractor, node_feature=args['node_feature'], device=device)


    # Load DAppSCAN dataset
    with open(args['dataset'], 'rb') as f:
        dataset = json.load(f)
    train_files = dataset[args['bugtype']]
    with open(args['valset'], 'rb') as f:
        dataset = json.load(f)
    val_files = dataset[args['bugtype']]  
    with open(args['testset'], 'rb') as f:
        dataset = json.load(f)
    test_files = dataset[args['bugtype']]

    
    print('Intersection: ')
    train_ids = get_node_ids(nx_graph, train_files)
    val_ids = get_node_ids(nx_graph, val_files)
    test_ids = get_node_ids(nx_graph, test_files)

    # assert intersection of 3 set train_ids, val_ids, test_ids is empty
    assert len(set(train_files) & set(val_files)) == 0
    assert len(set(train_files) & set(val_files)) == 0
    assert len(set(val_files) & set(test_files)) == 0


    assert set(train_ids) & set(val_ids) == set()
    assert set(train_ids) & set(test_ids) == set()
    assert set(val_ids) & set(test_ids) == set()

    targets = torch.tensor(model.node_labels, device=device)
    assert len(train_ids + val_ids + test_ids) == len(targets)
    buggy_node_ids = torch.nonzero(targets).squeeze().tolist()
    print('Buggy node {}/{} ({}%)'.format(len(set(buggy_node_ids)), len(targets), 100*len(set(buggy_node_ids))/len(targets)))
    # for fold, (train_ids, val_ids) in enumerate(kfold.split(total_train_ids)):
        # Init model 
    fold = 0
    model.reset_parameters()
    model.to(device)
    train_results[fold] = {'loss': [], 'acc': [], 'micro_f1': [], 'macro_f1': [], 'lrs': []}
    val_results[fold] = {'loss': [], 'acc': [], 'micro_f1': [], 'macro_f1': []}
    train_buggy_node_ids = set(buggy_node_ids).intersection(set(train_ids))
    print('Buggy nodes in train: {}/{} ({}%)'.format(len(train_buggy_node_ids), len(train_ids), 100*len(train_buggy_node_ids)/len(train_ids)))
    val_buggy_node_ids = set(buggy_node_ids).intersection(set(val_ids))
    print('Buggy nodes in valid: {}/{} ({}%)'.format(len(val_buggy_node_ids), len(val_ids), 100*len(val_buggy_node_ids)/len(val_ids)))
    test_buggy_node_ids =set(buggy_node_ids).intersection(set(test_ids))
    print('Buggy nodes in test: {}/{} ({}%)'.format(len(test_buggy_node_ids), len(test_ids), 100*len(test_buggy_node_ids)/len(test_ids)))
    print('Start training fold {} with {}/{} train/val smart contracts'.format(fold, len(train_ids), len(val_ids)))
    total_steps = epochs
    
    assert len(train_buggy_node_ids) + len(val_buggy_node_ids) + len(test_buggy_node_ids) == len(buggy_node_ids)

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, total_steps=total_steps)
    train_mask = get_binary_mask(number_of_nodes, train_ids)
    val_mask = get_binary_mask(number_of_nodes, val_ids)
    test_mask = get_binary_mask(number_of_nodes, test_ids)
    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()
    retain_graph = True if args['node_feature'] == 'han' else False
    for epoch in range(epochs):
        # print('Fold {} - Epochs {}'.format(fold, epoch))
        optimizer.zero_grad()
        logits = model()
        logits = logits.to(device)
        train_loss = loss_fcn(logits[train_mask], targets[train_mask]) 
        train_loss.backward(retain_graph=retain_graph)
        optimizer.step()
        scheduler.step()
        train_acc, train_micro_f1, train_macro_f1 = score(targets[train_mask], logits[train_mask])
        print('Train Loss: {:.4f} | Train Micro f1: {:.4f} | Train Macro f1: {:.4f} | Train Accuracy: {:.4f}'.format(
                train_loss.item(), train_micro_f1, train_macro_f1, train_acc))
        val_loss = loss_fcn(logits[val_mask], targets[val_mask]) 
        val_acc, val_micro_f1, val_macro_f1 = score(targets[val_mask], logits[val_mask])
        print('Val Loss:   {:.4f} | Val Micro f1:   {:.4f} | Val Macro f1:   {:.4f} | Val Accuracy:   {:.4f}'.format(
                val_loss.item(), val_micro_f1, val_macro_f1, val_acc))

        train_results[fold]['loss'].append(train_loss)
        train_results[fold]['micro_f1'].append(train_micro_f1)
        train_results[fold]['macro_f1'].append(train_macro_f1)
        train_results[fold]['acc'].append(train_acc)
        train_results[fold]['lrs'] += scheduler.get_last_lr()

        val_results[fold]['loss'].append(val_loss)
        val_results[fold]['micro_f1'].append(val_micro_f1)
        val_results[fold]['macro_f1'].append(val_macro_f1)
        val_results[fold]['acc'].append(val_acc)
    print('Saving model fold {}'.format(fold))
    dump_result(targets[val_mask], logits[val_mask], os.path.join(args['output_models'], f'confusion_{fold}.csv'))
    save_path = os.path.join(args['output_models'], f'hgt_fold_{fold}.pth')
    torch.save(model.state_dict(), save_path)
    print('Testing phase $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    bugtype = args['bugtype']
    print(f'Testing performance on {bugtype} smart contracts')
    print(f'Testing on {len(test_ids)} nodes')
    model.eval()
    with torch.no_grad():
        logits = model()
        logits = logits.to(device)
        test_acc, test_micro_f1, test_macro_f1 = score(targets[test_mask], logits[test_mask])
        print('Test Micro f1:   {:.4f} | Test Macro f1:   {:.4f} | Test Accuracy:   {:.4f}'.format(test_micro_f1, test_macro_f1, test_acc))
        print('Classification report', '\n', get_classification_report(targets[test_mask], logits[test_mask]))
        print('Confusion matrix', '\n', get_confusion_matrix(targets[test_mask], logits[test_mask]))
    return train_results, val_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MANDO Node Classifier')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    archive_params = parser.add_argument_group(title='Storage', description='Directories for util results')
    archive_params.add_argument('-ld', '--log_dir', type=str, default='./logs/node_classification', help='Directory for saving training logs and visualization')
    archive_params.add_argument('--output_models', type=str, default='./models/call_graph_rgcn',
                        help='Where you want to save your models')
    archive_params.add_argument('-e', '--num_epochs', type=int, default=100, help='Number of epochs')

    dataset_params = parser.add_argument_group(title='Dataset', description='Dataset paths')
    dataset_params.add_argument('--compressed_graph', type=str, default='./dataset/call_graph/compressed_graph/compress_call_graphs_no_solidity_calls.gpickle',
                        help='Compressed graphs of dataset which was extracted by graph helper tools')
    dataset_params.add_argument('--dataset', type=str, default='/Users/minh/Documents/2022/programing_analysis/tree-sitter-codeviews/datasets/DAppSCAN/DAppSCAN-source/dasp_categories/DASP_train_metadata.json',
                        help='Dicrectory of all souce code files which were used to extract the compressed graph')
    dataset_params.add_argument('--valset', type=str, default='/Users/minh/Documents/2022/programing_analysis/tree-sitter-codeviews/datasets/DAppSCAN/DAppSCAN-source/dasp_categories/DASP_val_metadata.json',
                        help='Dicrectory of all souce code files which is a partition of the dataset for testing')
    dataset_params.add_argument('--testset', type=str, default='/Users/minh/Documents/2022/programing_analysis/tree-sitter-codeviews/datasets/DAppSCAN/DAppSCAN-source/dasp_categories/DASP_test_metadata.json',
                        help='Dicrectory of all souce code files which is a partition of the dataset for testing')
    dataset_params.add_argument('--bugtype', type=str, default='access_control', help='The bug type you want to train on')

    node_feature_params = parser.add_argument_group(title='Node feature', description='Define the way to get node features')
    node_feature_params.add_argument('--feature_compressed_graph', type=str, default='./dataset/aggregate/compressed_graph/compressed_graphs.gpickle',
                        help='If "node_feature" is han, you mean use 2 HAN layers. The first one is HAN of CFGs as feature node for the second HAN of call graph, This is the compressed graphs were trained for the first HAN')
    node_feature_params.add_argument('--cfg_feature_extractor', type=str, default='./models/metapath2vec_cfg/han_fold_1.pth',
                        help='If "node_feature" is han, feature_extractor is a checkpoint of the first HAN layer')
    node_feature_params.add_argument('--feature_extractor', type=str, default='./models/metapath2vec_cfg/han_fold_1.pth',
                        help='If "node_feature" is "GAE" or "LINE" or "Node2vec", we need a extracted features from those models')
    node_feature_params.add_argument('--node_feature', type=str, default='metapath2vec',
                        help='Kind of node features we want to use, here is one of "nodetype", "metapath2vec", "han", "gae", "line", "node2vec"')
    
    train_option_params = parser.add_argument_group(title='Optional configures', description='Advanced options')
    train_option_params.add_argument('--k_folds', type=int, default=1, help='Config for cross validate strategy')
    train_option_params.add_argument('--test', action='store_true', help='Set true if you only want to run test phase')
    train_option_params.add_argument('--non_visualize', action='store_true',
                        help='Wheather you want to visualize the metrics')
    args = parser.parse_args().__dict__

    default_configure = {
    'lr': 0.0005,             # Learning rate
    'num_heads': 8,        # Number of attention heads for node-level attention
    'hidden_units': 8,
    'dropout': 0.6,
    'weight_decay': 0.001,
    'num_epochs': 100,
    'batch_size': 256,
    'patience': 100,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
    }
    args.update(default_configure)
    torch.manual_seed(args['seed'])

    if not os.path.exists(args['output_models']):
        os.makedirs(args['output_models'])

    # Training
    if not args['test']:
        print('Training phase =====================================================================================')
        print('====================================================================================================')
        train_results, val_results = main(args)
        if not args['non_visualize']:
            print('Visualizing')
            # visualize_average_k_folds(args, train_results, val_results)
    # Testing
    else:
        print('Testing phase')
        nx_graph = nx.read_gpickle(args['compressed_graph'])
        number_of_nodes = len(nx_graph)
        test_files = [f for f in os.listdir(args['testset']) if f.endswith('.sol')]
        model = HGTVulNodeClassifier(args['compressed_graph'], feature_extractor=None, node_feature=args['node_feature'], device=args['device'])
        model.load_state_dict(torch.load(args['feature_extractor'], map_location=torch.device('cpu')))
        model.eval()
        model.to(args['device'])
        test_ids = get_node_ids(nx_graph, test_files)
        targets = torch.tensor(model.node_labels, device=args['device'])
        buggy_node_ids = torch.nonzero(targets).squeeze().tolist()
        test_buggy_node_ids = set(buggy_node_ids) & set(test_ids)
        print('Buggy nodes in test: {}/{} ({}%)'.format(len(test_buggy_node_ids), len(test_ids), 100*len(test_buggy_node_ids)/len(test_ids)))
        test_mask = get_binary_mask(number_of_nodes, test_ids)
        if hasattr(torch, 'BoolTensor'):
            test_mask = test_mask.bool()
        print(f"Testing on {len(test_ids)} nodes")
        with torch.no_grad():
            logits = model()
            logits = logits.to(args['device'])
            print(torch.nonzero(targets, as_tuple=True)[0].shape)
            test_acc, test_micro_f1, test_macro_f1 = score(targets[test_mask], logits[test_mask])
            print('Test Micro f1:   {:.4f} | Test Macro f1:   {:.4f} | Test Accuracy:   {:.4f}'.format(test_micro_f1, test_macro_f1, test_acc))
            print('Classification report', '\n', get_classification_report(targets[test_mask], logits[test_mask]))
            print('Confusion matrix', '\n', get_confusion_matrix(targets[test_mask], logits[test_mask]))
