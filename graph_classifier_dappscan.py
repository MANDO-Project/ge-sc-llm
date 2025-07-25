import os
from shutil import rmtree
import random

import gc
import numpy as np
import torch
from tabulate import tabulate
from torch.nn.functional import cross_entropy
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import KFold, StratifiedKFold
from dgl.dataloading import GraphDataLoader

from sco_models.dataloader import EthIdsDataset
from sco_models.model_hgt import HGTVulGraphClassifier
from sco_models.visualization import visualize_average_k_folds, visualize_k_folds
from sco_models.utils import score, get_classification_report, get_confusion_matrix
from sco_models.tools import EarlyStopping


def train(args, model, train_loader, optimizer, loss_fcn, epoch):
    model.train()
    total_accucracy =  0
    total_macro_f1 = 0
    total_micro_f1 = 0
    total_loss = 0
    circle_lrs = []
    for idx, (batched_graph, labels) in enumerate(train_loader):
        labels = labels.to(args['device'])
        optimizer.zero_grad()
        logits, _ = model(batched_graph)
        loss = loss_fcn(logits, labels)
        train_acc, train_micro_f1, train_macro_f1, train_buggy_f1 = score(labels, logits)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)
        optimizer.step()
        total_accucracy += train_acc
        total_micro_f1 += train_micro_f1
        total_macro_f1 += train_macro_f1
        total_loss += loss.item()
        circle_lrs.append(optimizer.param_groups[0]["lr"])
    steps = idx + 1
    return model, total_loss/steps, total_micro_f1/steps, train_macro_f1/steps, total_accucracy/steps, train_buggy_f1/steps, circle_lrs


def validate(args, model, val_loader, loss_fcn):
    model.eval()
    total_loss = 0
    total_macro_f1 = 0
    total_micro_f1 = 0
    total_accucracy =  0
    with torch.no_grad():
        for idx, (batched_graph, labels) in enumerate(val_loader):
            labels = labels.to(args['device'])
            logits, _ = model(batched_graph)
            loss = loss_fcn(logits, labels)
            total_loss += loss.item()
            val_acc, val_micro_f1, val_macro_f1, val_buggy_f1 = score(labels, logits)
            total_accucracy += val_acc
            total_micro_f1 += val_micro_f1
            total_macro_f1 += val_macro_f1
    steps = idx + 1
    return total_loss/steps, total_micro_f1/steps, val_macro_f1/steps, total_accucracy/steps, val_buggy_f1/steps


def test(args, model, test_loader):
    model.eval()
    total_macro_f1 = 0
    total_micro_f1 = 0
    total_accucracy =  0
    total_logits = []
    total_target = []
    with torch.no_grad():
        for idx, (batched_graph, labels) in enumerate(test_loader):
            labels = labels.to(args['device'])
            # logits, _ = model(batched_graph, './forensics/graph_hiddens/reentrancy/creation_last_attention.pt')
            logits, _ = model(batched_graph)
            total_logits += logits.tolist()
            total_target += labels.tolist()
            test_acc, test_micro_f1, test_macro_f1, _ = score(labels, logits)
            total_accucracy += test_acc
            total_micro_f1 += test_micro_f1
            total_macro_f1 += test_macro_f1
    steps = idx + 1
    total_logits = torch.tensor(total_logits)
    total_target = torch.tensor(total_target)
    classification_report = get_classification_report(total_target, total_logits, output_dict=True)
    confusion_report = get_confusion_matrix(total_target, total_logits)
    return total_micro_f1/steps, test_macro_f1/steps, total_accucracy/steps, classification_report, confusion_report

def get_class_distribution(dataset, sample_ids):
    categories = {}
    for s in sample_ids:
        graph, label = dataset.__getitem__(s)
        label = int(label)
        if label in categories:
            categories[label].append(graph)
        else:
            categories[label] = [graph]
    return categories


def main(args):
    print('===============================================')
    print('Graph file ', args['compressed_graph'])
    epochs = args['num_epochs']
    k_folds = args['k_folds']
    device = args['device']
    ethdataset = EthIdsDataset(args['label'])
    # kfold = KFold(n_splits=k_folds, shuffle=True)
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)
    train_results = {}
    val_results = {'buggy_f1_folds': [], 'macro_f1_folds': []}
    # Get feature extractor
    print('Getting features')
    if args['node_feature'] == 'han':
        feature_extractor = HGTVulGraphClassifier(args['feature_compressed_graph'], node_feature='nodetype', hidden_size=16, device=args['device'])
        feature_extractor.load_state_dict(torch.load(args['feature_extractor']))
        feature_extractor.to(args['device'])
        feature_extractor.eval()
    else:
        feature_extractor = args['feature_extractor']

    classification_total_report = {'0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}, '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}, 'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}, 'weighted avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}}
    confusion_matrix_total_report = []
    test_ids = []
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    test_dataloader = GraphDataLoader(ethdataset, batch_size=args['batch_size'], drop_last=False, sampler=test_subsampler)
    total_train_ids = list(set(list(range(ethdataset.num_graphs))).difference(set(test_ids)))
    assert len(set(test_ids) & set(total_train_ids)) == 0
    for fold, (train_ids, val_ids) in enumerate(kfold.split(ethdataset.graphs, ethdataset.label)):
        train_categories = get_class_distribution(ethdataset, train_ids)
        val_categories = get_class_distribution(ethdataset, val_ids)
        print('---------------------------------------------------------------------------')
        print(f'Train class distribution: {len(train_categories[0])}/{len(train_categories[1])}')
        print(f'Val class distribution: {len(val_categories[0])}/{len(val_categories[1])}')
        assert len(test_ids) + len(train_ids) + len(val_ids) == len(ethdataset)
        assert len(set(test_ids) & set(train_ids)) == 0
        assert len(set(test_ids) & set(val_ids)) == 0
        train_results[fold] = {'loss': [], 'acc': [], 'micro_f1': [], 'macro_f1': [], 'buggy_f1': [], 'lrs': []}
        val_results[fold] = {'loss': [], 'acc': [], 'micro_f1': [], 'macro_f1': [], 'buggy_f1': []}
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        train_dataloader = GraphDataLoader(ethdataset,batch_size=args['batch_size'],drop_last=False,sampler=train_subsampler)
        val_dataloader = GraphDataLoader(ethdataset,batch_size=args['batch_size'],drop_last=False,sampler=val_subsampler)
        print('Start training fold {} with {}/{}/{} train/val/test smart contracts'.format(fold, len(train_subsampler), len(val_subsampler), len(test_ids)))
        total_steps = epochs
        model = HGTVulGraphClassifier(args['compressed_graph'], feature_extractor=feature_extractor, node_feature=args['node_feature'], device=device)
        model.reset_parameters()
        model.to(device)
        loss_fcn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=total_steps)
        early_stopping = EarlyStopping(patience=args['patience'], delta=0.0001, verbose=True)
        earliest_epoch = 0
        stop_epoch = epochs
        checkpoint_path = args['output_models']
        lrs = []
        for epoch in range(epochs):
            print('Fold {} - Epochss {}'.format(fold, epoch))
            model, train_loss, train_micro_f1, train_macro_f1, train_acc, train_buggy_f1, lrs = train(args, model, train_dataloader, optimizer, loss_fcn, epoch)
            print('Train Loss: {:.4f} | Train Micro f1: {:.4f} | Train Macro f1: {:.4f} | Train Accuracy: {:.4f}'.format(
                    train_loss, train_micro_f1, train_macro_f1, train_acc))
            val_loss, val_micro_f1, val_macro_f1, val_acc, val_buggy_f1 = validate(args, model, val_dataloader, loss_fcn)
            print('Val Loss:   {:.4f} | Val Micro f1:   {:.4f} | Val Macro f1:   {:.4f} | Val Accuracy:   {:.4f}'.format(
                    val_loss, val_micro_f1, val_macro_f1, val_acc))
            scheduler.step()
            train_results[fold]['loss'].append(train_loss)
            train_results[fold]['micro_f1'].append(train_micro_f1)
            train_results[fold]['macro_f1'].append(train_macro_f1)
            train_results[fold]['acc'].append(train_acc)
            train_results[fold]['buggy_f1'].append(train_buggy_f1)
            train_results[fold]['lrs'] += lrs

            val_results[fold]['loss'].append(val_loss)
            val_results[fold]['micro_f1'].append(val_micro_f1)
            val_results[fold]['macro_f1'].append(val_macro_f1)
            val_results[fold]['buggy_f1'].append(val_buggy_f1)
            val_results[fold]['acc'].append(val_acc)
            # Early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                if stop_epoch >= epoch:
                    checkpoint_path = args['output_models'].replace('.pth', f'_{epoch}_{fold}.pth')
                    early_stopping.save_checkpoint(val_loss, model, checkpoint_path)
                    earliest_epoch = epoch
                    break
                stop_epoch = epoch
            
        print("Early stopping at epoch {}".format(earliest_epoch))
        print('Saving model fold {}'.format(fold))

        # _, _, _, classification_report, confusion_report = test(args, model, val_dataloader)
        # for category, metrics in classification_total_report.items():
        #     for metric in metrics.keys():
        #         classification_total_report[category][metric].append(classification_report[category][metric])

        # confusion_matrix_total_report.append(confusion_report)

        print('Saving model fold {}'.format(fold))
        checkpoint_path = args['output_models'].replace('.pth', f'_{epoch}_{fold}.pth')
        torch.save(model.state_dict(), checkpoint_path)


        saved_models = model
        saved_models.load_state_dict(torch.load(checkpoint_path))
        saved_models.eval()
        _, _, _, classification_report, confusion_report = test(args, model, val_dataloader)
        
        val_results['buggy_f1_folds'].append(classification_report['1']['f1-score'])
        val_results['macro_f1_folds'].append(classification_report['macro avg']['f1-score'])
        print(confusion_report)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return train_results, val_results


def load_model(model_path):
    model = HGTVulGraphClassifier()
    model.load_state_dict(torch.load(model_path))
    return model.eval()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MANDO Graph Classifier')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    archive_params = parser.add_argument_group(title='Storage', description='Directories for util results')
    archive_params.add_argument('-ld', '--log_dir', type=str, default='./logs/graph_classification', help='Directory for saving training logs and visualization')
    archive_params.add_argument('--output_models', type=str, default='./models/call_graph', help='Where you want to save your models')
    dataset_params = parser.add_argument_group(title='Dataset', description='Dataset paths')
    dataset_params.add_argument('--compressed_graph', type=str, default='./dataset/call_graph/compressed_graph/compress_call_graphs_no_solidity_calls.gpickle', help='Compressed graphs of dataset which was extracted by graph helper tools')
    dataset_params.add_argument('--dataset', type=str, default='./dataset/aggregate/source_code', help='Dicrectory of all souce code files which were used to extract the compressed graph')
    dataset_params.add_argument('--testset', type=str, default='./dataset/smartbugs/source_code', help='Dicrectory of all souce code files which is a partition of the dataset for testing')
    dataset_params.add_argument('--label', type=str, default='./dataset/aggregate/labels.json', help='Label of sources in source code storage')
    dataset_params.add_argument('--checkpoint', type=str, default='./models/ijcai2020_smartbugs/han_fold_1.pth', help='Checkpoint of trained models')
    node_feature_params = parser.add_argument_group(title='Node feature', description='Define the way to get node features')
    node_feature_params.add_argument('--feature_extractor', type=str, default='./models/metapath2vec_cfg/han_fold_1.pth', help='If "node_feature" is "GAE" or "LINE" or "Node2vec", we need a extracted features from those models')
    node_feature_params.add_argument('--node_feature', type=str, default='metapath2vec', help='Kind of node features we want to use, here is one of "nodetype", "metapath2vec", "han", "gae", "line", "node2vec"')
    train_option_params = parser.add_argument_group(title='Optional configures', description='Advanced options')
    train_option_params.add_argument('--num_epochs', type=int, default=25, help='Config number of epochs')
    train_option_params.add_argument('--k_folds', type=int, default=5, help='Config for cross validate strategy')
    train_option_params.add_argument('--test', action='store_true', help='Set true if you only want to run test phase')
    train_option_params.add_argument('--non_visualize', action='store_true', help='Wheather you want to visualize the metrics')
    args = parser.parse_args().__dict__

    default_configure = {
    'lr': 0.001,             # Learning rate
    'num_heads': 8,        # Number of attention heads for node-level attention
    'hidden_units': 8,
    'dropout': 0.6,
    'weight_decay': 0.001,
    'num_epochs': 50,
    'batch_size': 1024,
    'patience': 25,
    'device': f'cuda:{random.randint(0, 1)}' if torch.cuda.is_available() else 'cpu'
    # 'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
    }
    args.update(default_configure)
    torch.manual_seed(args['seed'])

    os.makedirs(os.path.split(args['output_models'])[0], exist_ok=True)
    # if not os.path.exists(args['output_models']):
    #     os.makedirs(args['output_models'])

    # Training
    if not args['test']:
        print('Training phase')
        train_results, val_results = main(args)
        if not args['non_visualize']:
            print('Visualizing at: ', args['log_dir'])
            if os.path.exists(args['log_dir']):
                rmtree(args['log_dir'])
            visualize_average_k_folds(args, train_results, val_results)
            visualize_k_folds(args, train_results, val_results)
            
        print('Result of {} using {}'.format(args['compressed_graph'], args['node_feature']))
        print('Average Buggy F1 score of folds: {:.4f}'.format(np.mean(val_results['buggy_f1_folds'])*100))
        print('Average Macro F1 score of folds: {:.4f}'.format(np.mean(val_results['macro_f1_folds'])*100))
        
        print('Max Buggy F1 score of folds: {:.4f}'.format(np.max(val_results['buggy_f1_folds'])*100))
        print('Max Macro F1 score of folds: {:.4f}'.format(np.max(val_results['macro_f1_folds'])*100))
        
    # Testing
    else:
        print('Testing phase')
        for i in range(args['k_folds']):
            model = HGTVulGraphClassifier('/Users/minh/Documents/2022/smart_contract/mando/ge-sc-machine/sco/graphs/graph_detection/reentrancy_cfg_cg_compressed_graphs.gpickle', feature_extractor=args['feature_extractor'], node_feature=args['node_feature'], device=args['device'])
            model.load_state_dict(torch.load('/Users/minh/Documents/2022/smart_contract/mando/ge-sc-machine/sco/models/graph_detection/nodetype/reentrancy_hgt.pth'))
            model.to(args['device'])
            model.eval()
            # test_micro_f1, test_macro_f1, test_acc = test(args, model, test_dataloader)
            # print('Test Micro f1:   {:.4f} | Test Macro f1:   {:.4f} | Test Accuracy:   {:.4f}'.format(test_micro_f1, test_macro_f1, test_acc))
