from __future__ import division, print_function

import argparse
import json
import os
import random
import time
from typing import List, Dict
from pathlib import Path
from datetime import datetime
import pytz

# import tensorflow as tf
import torch
import torch.nn as nn
from sklearn import metrics
import deepdish as dd
from tqdm import trange

from models_pytorch import TGCN
# from torch_geometric.data.sampler import NeighborSampler
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--no_sparse', action='store_true')
    parser.add_argument("--load_ckpt", action='store_true')
    parser.add_argument('--featureless', action='store_true')
    parser.add_argument('--use_sem', action='store_true')
    parser.add_argument('--use_syn', action='store_true')
    parser.add_argument('--use_seq', action='store_true')
    parser.add_argument("--save_path", type=str, default='./saved_model', help="the path of saved model")
    parser.add_argument('--dataset', type=str, default='mr', help='dataset name, default to mr')
    parser.add_argument('--model', type=str, default='gcn', help='model name, default to gcn')
    parser.add_argument('--lr', '--learning_rate', default=0.00002, type=float)   # 0.002/0.0002
    parser.add_argument("--epochs", default=300, type=int) 
    parser.add_argument("--hidden", default=200, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.8, type=float)   # 0.5/0.3/0.1
    parser.add_argument("--weight_decay", default=0.000001, type=float)
    parser.add_argument("--early_stop", default=300, type=int)
    parser.add_argument("--max_degree", default=3, type=int)
    parser.add_argument("--model_name", default='model', type=str)
    parser.add_argument("--run_id", default=None, type=str)
    return parser.parse_args(args)



def save_model(model, optimizer, args, timestamp):
    '''
    Save the parameters of the model   the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    if not os.path.exists(os.path.join(args.save_path,'run_{}'.format(timestamp))):
        logger.info(os.path.join(args.save_path,'run_{}'.format(timestamp)))
        os.makedirs(os.path.join(args.save_path,'run_{}'.format(timestamp)))
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'run_{}/{}_config.json'.format(timestamp, args.model_name)), 'w') as fjson:
        json.dump(argparse_dict, fjson)
    logger.info("Configurations for training:")
    logger.debug(argparse_dict)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}
    torch.save(save_dict, os.path.join(args.save_path, 'run_{}/{}.bin'.format(timestamp, args.model_name)))

    return True

def train(args, features, train_label, train_mask, val_label, val_mask, test_label, test_mask, model, indice_list, weight_list)-> List[Dict]:
    cost_train = []
    cost_valid = []
    cost_test = []
    acc_train = []
    acc_valid = []
    acc_test = []

    max_acc = 0.0
    min_cost = 10.0
    # for (name, param) in model.named_parameters():
    #     print(name)
    # weight_decay_list = (param for (name, param) in model.named_parameters() if 'layers.0' in name)
    # no_decay_list = (param for (name, param) in model.named_parameters() if 'layers.0' not in name)
    # parameters = [{'params':weight_decay_list},{'params':no_decay_list, 'weight_decay':0.0}]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    model.train()

    store_time = []

    for epoch in range(args.epochs):
        
        t = time.time()
        # Construct feed dictionary
        # feed_dict = construct_feed_dict(
        #     features, support, support_mix, y_train, train_mask, placeholders)
        # feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = model(features, indice_list, weight_list, 1-args.dropout)
        pre_loss = loss_fct(outs, train_label)
        train_pred = torch.argmax(outs, dim=-1)
        ce_loss = (pre_loss * train_mask/train_mask.mean()).mean()
        train_acc = ((train_pred == train_label).float() * train_mask/train_mask.mean()).mean()
        # loss = ce_loss + tmp_loss
        loss = ce_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        model.eval()
        # Validation
        valid_cost, valid_acc, pred, labels, duration = evaluate(args,
            features, val_label, val_mask, model, indice_list, weight_list)

        # Testing
        test_cost, test_acc, pred, labels, test_duration = evaluate(args,
            features, test_label, test_mask, model, indice_list, weight_list)
        model.train()

        cost_valid.append(valid_cost)

        cost_train.append(loss.item())
        cost_test.append(test_cost)
        acc_train.append(train_acc.item())
        acc_valid.append(valid_acc)
        acc_test.append(test_acc)

        # time taken to perform training for 1 epoch
        epoch_time = time.time() - t

        # logger.info("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "train_acc=",
        #     "{:.5f}".format(train_acc.item()), "val_loss=", "{:.5f}".format(valid_cost),
        #     "val_acc=", "{:.5f}".format(valid_acc), "test_loss=", "{:.5f}".format(test_cost), "test_acc=",
        #     "{:.5f}".format(test_acc), "time=", "{:.5f}".format(time.time() - t))
        logger.info("\n Epoch: {:04d} train_loss= {:.5f} train_acc= {:.5f} val_loss= {:.5f} val_acc= {:.5f} test_loss= {:.5f} test_acc= {:.5f} time= {:.5f}".format(
        epoch + 1, loss.item(), train_acc.item(), valid_cost, valid_acc, test_cost, test_acc, epoch_time))

        store_time.append(epoch_time) # append to list of epoch time runs

        # save model
        # if epoch > 700 and cost_valid[-1] < min_cost:
        if cost_valid[-1] < min_cost:
            saved_res = save_model(model, optimizer, args, timestamp)
            min_cost = cost_valid[-1]
            logger.info("Current best loss {:.5f}".format(min_cost))
        
        else:
            saved_res = False
        
        # if acc_valid[-1] > max_acc:
        #     save_model(model, optimizer, args)
        #     min_cost = cost_valid[-1]
        #     max_acc = acc_valid[-1]
        #     print("Current best acc {:.5f}".format(max_acc))

        # early stoppage implementation
        # training loop terminates if validation cost exceeds mean of previous epochs 
        if epoch > args.early_stop and cost_valid[-1] > np.mean(cost_valid[-(args.early_stop + 1):-1]):
            logger.info("Early stopping...")
            break

    # pickle store time 
    pickle_artifact(
        args=args,
        filename='epoch_time',
        timestamp=timestamp,
        artifact=store_time
        )

    if not saved_res:
        save_model(model, optimizer, args, timestamp)
    logger.info("Optimization Finished!")

    loss_results = {
        'train_loss': cost_train,
        'valid_loss': cost_valid,
        'test_loss': cost_test
    }

    acc_results = {
        'train_acc': acc_train,
        'valid_acc': acc_valid,
        'test_acc': acc_test
    }

    return [loss_results, acc_results]

def evaluate(args, features, label, mask, model, indice_list, weight_list):
    t_test = time.time()
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        outs = model(features, indice_list, weight_list, 1)
        pre_loss = loss_fct(outs, label)
        pred = torch.argmax(outs, dim=-1)
        ce_loss = (pre_loss * mask/mask.mean()).mean()
        loss = ce_loss
        acc = ((pred == label).float() * mask/mask.mean()).mean()
    # feed_dict_val = construct_feed_dict(
    #     features, support, support_mix, labels, mask, placeholders)
    # outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    return loss.item(), acc.item(), pred.cpu().numpy(), label.cpu().numpy(), (time.time() - t_test)

def load_ckpt(model):
    model_dict = model.state_dict()
    pretrained_dict = dd.io.load('./gcn.h5')
    model_dict['layers.0.intra_convs.0'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_1_vars_weights_0:0'].T, dtype=torch.float)
    model_dict['layers.0.inter_convs.0'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_1_vars_weights_00:0'].T, dtype=torch.float)
    model_dict['layers.0.intra_convs.1'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_1_vars_weights_1:0'].T, dtype=torch.float)
    model_dict['layers.0.inter_convs.1'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_1_vars_weights_11:0'].T, dtype=torch.float)
    model_dict['layers.0.intra_convs.2'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_1_vars_weights_2:0'].T, dtype=torch.float)
    model_dict['layers.0.inter_convs.2'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_1_vars_weights_22:0'].T, dtype=torch.float)
    model_dict['layers.1.intra_convs.0'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_2_vars_weights_0:0'].T, dtype=torch.float)
    model_dict['layers.1.inter_convs.0'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_2_vars_weights_00:0'].T, dtype=torch.float)
    model_dict['layers.1.intra_convs.1'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_2_vars_weights_1:0'].T, dtype=torch.float)
    model_dict['layers.1.inter_convs.1'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_2_vars_weights_11:0'].T, dtype=torch.float)
    model_dict['layers.1.intra_convs.2'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_2_vars_weights_2:0'].T, dtype=torch.float)
    model_dict['layers.1.inter_convs.2'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_2_vars_weights_22:0'].T, dtype=torch.float)    
    model.load_state_dict(model_dict)

# tf.compat.v1.disable_eager_execution()
def get_edge_tensor(adj):
    row = torch.tensor(adj.row, dtype=torch.long)
    col = torch.tensor(adj.col, dtype=torch.long)
    data = torch.tensor(adj.data, dtype=torch.float)
    indice = torch.stack((row,col),dim=0)
    return indice, data


def main(args, timestamp):
    start_time = time.time()
    
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info("Training is running on {}".format(device))
    else:
        device = None
        logger.info("Training is running on CPU")
    
    # Set random seed
    seed=147
    logger.info("Seed used: {}".format(seed))    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
    # Load data
    # adj, adj1, adj2, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, val_size,test_size, num_labels = load_corpus_torch(args, device)
    adj_lst, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, val_size,test_size, num_labels = load_corpus_torch(args, device)
    # for adj in adj_lst:
    #     adj = adj.tocoo()
    # adj = adj.tocoo()
    # adj1 = adj1.tocoo()
    # adj2 = adj2.tocoo()
    logger.debug("adj:\n {}".format(adj_lst[0]))

    logger.info("The shape of adj is {}".format(adj_lst[0].shape))

    # one-hot features
    # features = torch.eye(adj.shape[0], dtype=torch.float).to_sparse().to(device)
    # support_mix = [adj, adj1, adj2]
    # support_mix = adj_lst
    indice_list, weight_list = [] , []
    for adjacency in adj_lst:
        adjacency = adjacency.tocoo()
        ind, dat = get_edge_tensor(adjacency)
        indice_list.append(ind.to(device))
        weight_list.append(dat.to(device))
        
    in_dim = adj_lst[0].shape[0]
    model = TGCN(in_dim=in_dim, hidden_dim=args.hidden, out_dim=num_labels, num_graphs=len(adj_lst), dropout=args.dropout, n_layers=args.layers, bias=False, featureless=args.featureless)
    features = torch.tensor(list(range(in_dim)), dtype=torch.long).to(device)
    
    model.to(device)
    
    if args.do_train:
        logger.info("Starting training")
        results = train(args, features, y_train, train_mask, y_val, val_mask, y_test, test_mask, model, indice_list, weight_list)

        with open(os.path.join(args.save_path ,'run_{}/{}_train_results.pkl'.format(timestamp, args.model_name)), 'wb') as f:
            pkl.dump(results, f)

        logger.info("Successfully pickled file '{}_train_results.pkl' with loss and accuracy metrics to {}".format(args.model_name, args.save_path))

    if args.do_valid:
        logger.info("Starting validation")
        # FLAGS.dropout = 1.0
        save_dict = torch.load(os.path.join(args.save_path, 'run_{}/{}.bin'.format(timestamp, args.model_name)))
        if args.load_ckpt:
            load_ckpt(model)
        else:
            model.load_state_dict(save_dict['model_state_dict'])
        model.eval()
        # Testing
        val_cost, val_acc, pred, labels, val_duration = evaluate(args,
            features, y_val, val_mask, model, indice_list, weight_list)
        # logger.info("Val set results:", "cost=", "{:.5f}".format(val_cost),
        #     "accuracy=", "{:.5f}".format(val_acc), "time=", "{:.5f}".format(val_duration))
        logger.info("Val set results: cost= {:.5f} accuracy= {:.5f} time= {:.5f}".format(val_cost, val_acc, val_duration))

        val_pred = []
        val_labels = []
        logger.debug(val_mask)
        logger.debug(len(val_mask))
        for i in range(len(val_mask)):
            if val_mask[i] == 1:
                val_pred.append(pred[i])
                val_labels.append(labels[i])

        logger.info("Val Precision, Recall and F1-Score...")
        logger.info("\n {} ".format(metrics.classification_report(val_labels, val_pred, digits=4)))
        logger.info("Macro average Val Precision, Recall and F1-Score...")
        logger.info(metrics.precision_recall_fscore_support(val_labels, val_pred, average='macro'))
        logger.info("Micro average Val Precision, Recall and F1-Score...")
        logger.info(metrics.precision_recall_fscore_support(val_labels, val_pred, average='micro'))

    if args.do_test:
        # FLAGS.dropout = 1.0
        logger.info("Starting testing")
        save_dict = torch.load(os.path.join(args.save_path, 'run_{}/{}.bin'.format(timestamp, args.model_name)))
        if args.load_ckpt:
            load_ckpt(model)
        else:
            model.load_state_dict(save_dict['model_state_dict'])
        model.eval()
        # Testing
        test_cost, test_acc, pred, labels, test_duration = evaluate(args,
            features, y_test, test_mask, model, indice_list, weight_list)
        # logger.info("Test set results:", "cost=", "{:.5f}".format(test_cost),
        #     "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
        logger.info("Test set results: cost= {:.5f} accuracy= {:.5f} time= {:.5f}".format(test_cost, test_acc, test_duration))

        test_pred = []
        test_labels = []
        logger.debug("Test mask:")
        logger.debug(len(test_mask))
        for i in range(len(test_mask)):
            if test_mask[i] == 1:
                test_pred.append(pred[i])
                test_labels.append(labels[i])

        logger.info("Test Precision, Recall and F1-Score...")
        logger.info('\n {}'.format(metrics.classification_report(test_labels, test_pred, digits=4)))
        logger.info("Macro average Test Precision, Recall and F1-Score...")
        logger.info(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
        logger.info("Micro average Test Precision, Recall and F1-Score...")
        logger.info(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))
    end_time = time.time()
    logger.info("Total execution time: {} seconds".format(round(end_time-start_time,2)))
        
if __name__ == '__main__':
    
    # retrieve execution timestamp for logs
    sgt = pytz.timezone('Asia/Singapore')
    timestamp = datetime.now(sgt).strftime("%Y-%m-%d_%H-%M-%S")

    # set up logging
    log_path = os.path.join(Path(os.path.abspath(os.path.dirname(__file__)), '../logs'))
    logger = setup_logging(log_path=log_path, log_name='training_log', timestamp=timestamp)

    main(parse_args(), timestamp)