import os
from time import time


from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import json
import gc
import numpy as np
import networkx as nx
import torch
from torch import nn
from transformers import T5EncoderModel
from transformers import T5TokenizerFast

from transformers import AutoTokenizer, T5ForConditionalGeneration

from sco_models.tools import EarlyStopping
from sco_models.utils import score, get_confusion_matrix


def generate_coarse_grained_data(nx_graph, label_file):
    annotations = {}
    with open(label_file, 'r') as f:
        _annotations = json.load(f)
    for ann in _annotations:
        annotations[ann['contract_name']] = ann['targets']
    x_token = {}
    for _, node in nx_graph.nodes(data=True):
        # node_emb = node['token']
        node_emb = torch.tensor(node['token']).unsqueeze(0)
        sm = node['source_file']
        if sm not in x_token:
            x_token[sm] = node_emb
        else:
            x_token[sm] = torch.cat((x_token[sm], node_emb), dim=0)
    
    x_data = None
    y_data = None
    for k, v in x_token.items():
        is_vul = int(annotations[k]) if k in annotations else 0
        if x_data is None and y_data is None:
                x_data = torch.mean(v, dim=0).unsqueeze(0)
                y_data = torch.tensor([is_vul])
        else:
            x_data = torch.cat((x_data, torch.mean(v, dim=0).unsqueeze(0)), dim=0)
            y_data = torch.cat((y_data, torch.tensor([is_vul])), dim=0)
    return x_data, y_data
        

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    bug_type = {
              'access_control': 57,
              'arithmetic': 60, 
              'denial_of_service': 46,
              'front_running': 44,
              'reentrancy': 71, 
              'time_manipulation': 50, 
              'unchecked_low_level_calls': 95
              }
    
    # MODEL = 'solidity-t5'
    # MODEL = 'starcoder'
    # MODEL = 'solidity-t5-tunned'
    # MODEL = 'codet5p-770m'
    MODEL = 'LINE'
    # MODEL_URL = 'Salesforce/codet5p-770m'
    # tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_URL}", use_auth_token=True, trust_remote_code=True)
    # model = T5EncoderModel.from_pretrained(f"{MODEL_URL}", use_auth_token=True).to(device)
    K_FOLDS = 10
    EPOCHS = 100
    PATIENCE = 4
    DELTA = 0.001
    GRAPH_TYPE = 'cfg_cg'
    FEATURE = 'token'
    MAX_INPUT_LENGTH = 516
    LR = 0.005

    for bug, counter in bug_type.items():
        print(bug, counter)
        output_models = f'experiments/models/node_classificiation/hgt/{GRAPH_TYPE}/GCN/{MODEL}/{bug}/'
        # output_models = f'experiments/models/graph_classificiation/hgt/{GRAPH_TYPE}/GCN/{MODEL}/{bug}/'
        os.makedirs(output_models, exist_ok=True)
        kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True)
        # gpickle_file = f'experiments/ge-sc-data/source_code/{bug}/{GRAPH_TYPE}/encode_token_{MODEL}_{GRAPH_TYPE}_compressed_graphs.gpickle'
        # gpickle_file = f'experiments/ge-sc-data/source_code/{bug}/{GRAPH_TYPE}/encode_coarse_grained_token_{MODEL}_{GRAPH_TYPE}_compressed_graphs.gpickle'
        
        
        
        # Node level
        gpickle_file = f'experiments/ge-sc-data/source_code/{bug}/{GRAPH_TYPE}/{MODEL}_{GRAPH_TYPE}_compressed_graphs.gpickle'
        print(gpickle_file)
        nx_graph = nx.read_gpickle(gpickle_file)
        x_data = None
        y_data = None
        for _node in list(nx_graph.nodes(data=True)):
            n_id = _node[0]
            node = _node[1]
            is_vul = 1 if node['node_info_vulnerabilities'] else 0
            # input_ids = tokenizer(
            #     node['code'], return_tensors="pt", max_length=MAX_INPUT_LENGTH, padding="max_length", truncation=True
            # ).input_ids.to(device)
            # outputs = model(input_ids=input_ids)
            # last_hidden_states = outputs.last_hidden_state.detach()
            node_emb = torch.tensor(node['token'])
            last_hidden_states = node_emb.unsqueeze(0)
            # print(last_hidden_states.shape)
            if x_data is None and y_data is None:
                x_data = last_hidden_states
                # x_data = torch.mean(last_hidden_states, dim=1)
                y_data = torch.tensor([is_vul])
            else:
                x_data = torch.cat((x_data, last_hidden_states), dim=0)
                # x_data = torch.cat((x_data, torch.mean(last_hidden_states, dim=1)), dim=0)
                y_data = torch.cat((y_data, torch.tensor([is_vul])), dim=0)
        print(torch.sum(y_data)/y_data.shape[0])
        # torch.save(x_data, 'token_' + MODEL_URL.replace('/', '_') + '.pt')
        # torch.save(y_data, 'label_' + MODEL_URL.replace('/', '_') + '.pt')
        # torch.save(x_data, 'token_' + gpickle_file.replace('.gpickle', '.pt'))
        # torch.save(y_data, 'label_' + gpickle_file.replace('.gpickle', '.pt'))

        # # Graph level
        # label_file = f'experiments/ge-sc-data/source_code/{bug}/{GRAPH_TYPE}/graph_labels.json'
        # gpickle_file = f'experiments/ge-sc-data/source_code/{bug}/{GRAPH_TYPE}/{MODEL}_coarse_grained_{GRAPH_TYPE}_compressed_graphs.gpickle'
        # print(gpickle_file)
        # nx_graph = nx.read_gpickle(gpickle_file)
        # x_data, y_data = generate_coarse_grained_data(nx_graph, label_file)
        # print(torch.sum(y_data)/y_data.shape[0])

        # source_embedding_file = f'experiments/ge-sc-data/source_code/{bug}/{GRAPH_TYPE}/encode_coarse_grained_token_{MODEL}.pt'
        # print(source_embedding_file)
        # dataset = torch.load(source_embedding_file)
        # x_data, y_data = dataset['source_embedding'], dataset['label']
    
        hidden_size = x_data.shape[-1]
        print(x_data.shape)
        print(y_data.shape)
        val_results = {'buggy_f1_folds': [], 'macro_f1_folds': []}
        for fold, (train_ids, val_ids) in enumerate(kfold.split(x_data, y_data)):
            print('Training fold {}'.format(fold))
            train_token = x_data[train_ids].to(device)
            train_target = y_data[train_ids].to(device)
            val_token = x_data[val_ids].to(device)
            val_target = y_data[val_ids].to(device)

            classifier = torch.nn.Linear(hidden_size, 2)
            classifier.to(device)
            classifier.train()
            loss_fcn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=EPOCHS)
            early_stopping = EarlyStopping(patience=PATIENCE, delta=DELTA, verbose=True)

            stop_epoch = EPOCHS
            earliest_epoch = 0
            checkpoint_path = os.path.join(output_models, f'hgt_{EPOCHS}_{fold}.pth')
            for epoch in range(EPOCHS):
                optimizer.zero_grad()
                train_logits = classifier(train_token)
                train_loss = loss_fcn(train_logits, train_target)
                train_logits = train_logits.to(device)
                train_loss.backward()
                optimizer.step()
                scheduler.step()
                with torch.no_grad():
                    val_logits = classifier(val_token)
                    val_loss = loss_fcn(val_logits, val_target)
                # Early stopping
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    if stop_epoch >= epoch:
                        checkpoint_path = os.path.join(output_models, f'hgt_{epoch}_{fold}.pth')
                        early_stopping.save_checkpoint(val_loss, classifier, checkpoint_path)
                        earliest_epoch = epoch
                        break
                    stop_epoch = epoch
            # torch.save(classifier.state_dict(), checkpoint_path)
            print("Early stopping at epoch {}".format(earliest_epoch))
            print('Saving model fold {}'.format(fold))
            # torch.save(model.state_dict(), os.path.join(output_models, f'hgt_{epoch}_{fold}.pth'))

            saved_classifier = classifier
            saved_classifier.load_state_dict(torch.load(checkpoint_path))
            saved_classifier.eval()
            with torch.no_grad():
                logits = saved_classifier(val_token)
                logits = logits.to(device)
                val_acc, val_micro_f1, val_macro_f1, val_buggy_f1 = score(val_target, logits)
            print(get_confusion_matrix(val_target.cpu(), logits.cpu()))
            val_results['buggy_f1_folds'].append(val_buggy_f1)
            val_results['macro_f1_folds'].append(val_macro_f1)
            
            del classifier
            gc.collect()
            torch.cuda.empty_cache()

        print('Average Buggy F1 score of folds: {:.4f}'.format(np.mean(val_results['buggy_f1_folds'])*100))
        print('Average Macro F1 score of folds: {:.4f}'.format(np.mean(val_results['macro_f1_folds'])*100))