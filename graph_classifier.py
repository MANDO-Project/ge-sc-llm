import os
from shutil import rmtree

import gc
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from sco_models.utils import score, get_classification_report, get_confusion_matrix
from sco_models.tools import EarlyStopping
from sco_models.classifier_cli import (
    build_graph_parser,
    ensure_output_parent,
    finalize_graph_args,
    validate_paths,
)


def train(args, model, train_loader, optimizer, loss_fcn, epoch):
    model.train()
    total_accucracy =  0
    total_macro_f1 = 0
    total_micro_f1 = 0
    total_buggy_f1 = 0
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
        total_buggy_f1 += train_buggy_f1
        total_loss += loss.item()
        circle_lrs.append(optimizer.param_groups[0]["lr"])
    steps = idx + 1
    return model, total_loss/steps, total_micro_f1/steps, total_macro_f1/steps, total_accucracy/steps, total_buggy_f1/steps, circle_lrs


def validate(args, model, val_loader, loss_fcn):
    model.eval()
    total_loss = 0
    total_macro_f1 = 0
    total_micro_f1 = 0
    total_accucracy =  0
    total_buggy_f1 = 0
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
            total_buggy_f1 += val_buggy_f1
    steps = idx + 1
    return total_loss/steps, total_micro_f1/steps, total_macro_f1/steps, total_accucracy/steps, total_buggy_f1/steps


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
    return total_micro_f1/steps, total_macro_f1/steps, total_accucracy/steps, classification_report, confusion_report

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
    from dgl.dataloading import GraphDataLoader

    from sco_models.dataloader import EthIdsDataset
    from sco_models.model_hgt import HGTVulGraphClassifier

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

    # dataloader =  GraphDataLoader(ethdataset)
    # # for graphs, labels in dataloader:
    # for epoch in range(epochs):
    classification_total_report = {'0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}, '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}, 'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}, 'weighted avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}}
    confusion_matrix_total_report = []
    # test_ids = [ethdataset.filename_mapping[sc] for sc in os.listdir(args['testset']) if sc.endswith('.sol')]
    test_ids = []
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    test_dataloader = GraphDataLoader(ethdataset, batch_size=args['batch_size'], drop_last=False, sampler=test_subsampler)
    total_train_ids = list(set(list(range(ethdataset.num_graphs))).difference(set(test_ids)))
    assert len(set(test_ids) & set(total_train_ids)) == 0
    # for fold, (train_ids, val_ids) in enumerate(kfold.split(total_train_ids)):
    for fold, (train_ids, val_ids) in enumerate(kfold.split(ethdataset.graphs, ethdataset.label)):
        # train_ids = [total_train_ids[i] for i in _train_ids]
        # print(_train_ids == np.array(train_ids))
        # val_ids = [total_train_ids[i] for i in _val_ids]
        # print(_val_ids == np.array(val_ids))
        train_categories = get_class_distribution(ethdataset, train_ids)
        val_categories = get_class_distribution(ethdataset, val_ids)
        print('---------------------------------------------------------------------------')
        print([ethdataset.graphs[i] for i in train_ids if int(ethdataset.label[i]) == 0])
        print('---------------------------------------------------------------------------')
        print([ethdataset.graphs[i] for i in train_ids if int(ethdataset.label[i]) == 1])
        print('---------------------------------------------------------------------------')
        print([ethdataset.graphs[i] for i in val_ids if int(ethdataset.label[i]) == 0])
        print('---------------------------------------------------------------------------')
        print([ethdataset.graphs[i] for i in val_ids if int(ethdataset.label[i]) == 1])
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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.000005)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=total_steps)
        early_stopping = EarlyStopping(patience=args['patience'], delta=0.005, verbose=True)
        earliest_epoch = 0
        stop_epoch = epochs
        checkpoint_path = args['output_models']
        lrs = []
        for epoch in range(epochs):
            # print('Fold {} - Epochs {}'.format(fold, epoch))
            model, train_loss, train_micro_f1, train_macro_f1, train_acc, train_buggy_f1, lrs = train(args, model, train_dataloader, optimizer, loss_fcn, epoch)
            # print('Train Loss: {:.4f} | Train Micro f1: {:.4f} | Train Macro f1: {:.4f} | Train Accuracy: {:.4f}'.format(
            #         train_loss, train_micro_f1, train_macro_f1, train_acc))
            val_loss, val_micro_f1, val_macro_f1, val_acc, val_buggy_f1 = validate(args, model, val_dataloader, loss_fcn)
            # print('Val Loss:   {:.4f} | Val Micro f1:   {:.4f} | Val Macro f1:   {:.4f} | Val Accuracy:   {:.4f}'.format(
                    # val_loss, val_micro_f1, val_macro_f1, val_acc))
            # scheduler.step()
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

        # print('Saving model fold {}'.format(fold))
        # save_path = os.path.join(args['output_models'], f'han_fold_{fold}.pth')
        # bugtype = args['log_dir'].split('/')[-1]
        # torch.save(model.state_dict(), save_path)

        torch.save(model.state_dict(), args['output_models'])
        # print('Testing phase')
        # print(f'Testing on {len(test_ids)} nodes')

        saved_models = model
        saved_models.load_state_dict(torch.load(checkpoint_path))
        saved_models.eval()
        _, _, _, classification_report, confusion_report = test(args, model, val_dataloader)
        
        val_results['buggy_f1_folds'].append(classification_report['1']['f1-score'])
        val_results['macro_f1_folds'].append(classification_report['macro avg']['f1-score'])
        print(confusion_report)

    
    # headers = ['precision', 'recall', 'f1-score', 'avg_support']
    # classification_tabular_report = []
    # for category, metrics in classification_total_report.items():
    #     row = [category]
    #     for metric in metrics.keys():
    #         std = np.std(classification_total_report[category][metric])
    #         classification_total_report[category][metric] = np.max(classification_total_report[category][metric])
    #         row.append(f'{classification_total_report[category][metric]}(#{classification_total_report[category][metric]*std:.2f})')
    #     classification_tabular_report.append(row)
    # print(tabulate(classification_tabular_report, headers=headers))
    # print(np.round(np.mean(confusion_matrix_total_report, axis=0)))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return train_results, val_results


def load_model(model_path):
    from sco_models.model_hgt import HGTVulGraphClassifier

    model = HGTVulGraphClassifier()
    model.load_state_dict(torch.load(model_path))
    return model.eval()


def _graph_test_ids(dataset, testset):
    if not testset or not os.path.isdir(testset):
        return list(range(len(dataset)))
    test_names = {f for f in os.listdir(testset) if f.endswith('.sol')}
    if not test_names:
        return list(range(len(dataset)))
    selected = []
    for idx, graph_name in enumerate(dataset.graphs):
        if graph_name in test_names or os.path.basename(graph_name) in test_names:
            selected.append(idx)
    return selected or list(range(len(dataset)))


def run_inference(args):
    from dgl.dataloading import GraphDataLoader

    from sco_models.dataloader import EthIdsDataset
    from sco_models.model_hgt import HGTVulGraphClassifier

    validate_paths(args, ['compressed_graph', 'checkpoint', 'label'])
    ethdataset = EthIdsDataset(args['label'])
    test_ids = _graph_test_ids(ethdataset, args.get('testset'))
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    test_dataloader = GraphDataLoader(
        ethdataset,
        batch_size=args['batch_size'],
        drop_last=False,
        sampler=test_subsampler,
    )
    model = HGTVulGraphClassifier(
        args['compressed_graph'],
        feature_extractor=args['feature_extractor'],
        node_feature=args['node_feature'],
        device=args['device'],
    )
    model.load_state_dict(torch.load(args['checkpoint'], map_location=args['device']))
    model.to(args['device'])
    model.eval()
    test_micro_f1, test_macro_f1, test_acc, report, confusion = test(args, model, test_dataloader)
    print(f'Testing on {len(test_ids)} smart contracts')
    print('Test Micro f1:   {:.4f} | Test Macro f1:   {:.4f} | Test Accuracy:   {:.4f}'.format(test_micro_f1, test_macro_f1, test_acc))
    print('Classification report', '\n', report)
    print('Confusion matrix', '\n', confusion)


if __name__ == '__main__':
    parser = build_graph_parser()
    args = finalize_graph_args(parser.parse_args())
    torch.manual_seed(args['seed'])
    ensure_output_parent(args['output_models'])

    # Training
    if not args['test']:
        print('Training phase')
        validate_paths(args, ['compressed_graph', 'label'])
        train_results, val_results = main(args)
        if not args['non_visualize']:
            print('Visualizing')
            if os.path.exists(args['log_dir']):
                rmtree(args['log_dir'])
            # visualize_average_k_folds(args, train_results, val_results)
            # visualize_k_folds(args, train_results, val_results)
        print('Average Buggy F1 score of folds: {:.4f}'.format(np.mean(val_results['buggy_f1_folds'])*100))
        print('Average Macro F1 score of folds: {:.4f}'.format(np.mean(val_results['macro_f1_folds'])*100))
    # Testing
    else:
        print('Testing phase')
        run_inference(args)
