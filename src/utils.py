import numpy as np
import pickle as pkl
import scipy.sparse as sp
import os
import torch
import json
import random
import time
import logging
from logging.handlers import RotatingFileHandler
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def plot_loss_acc(train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, fig_name, header):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    x = np.arange(len(train_loss))
    max_loss = max(max(train_loss), max(val_loss), max(test_loss))  # Include test_loss in max calculation

    fig, ax1 = plt.subplots(figsize=(10, 6))  # Increase the size for better readability
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')  # Add color to the loss axis
    ax1.set_ylim([0, max_loss + 1])
    lns1 = ax1.plot(x, train_loss, 'b-', marker='o', label='Train Loss')  # Use solid lines and markers
    lns2 = ax1.plot(x, val_loss, 'g-', marker='o', label='Validation Loss')
    lns3 = ax1.plot(x, test_loss, 'r-', marker='o', label='Test Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()  # Create a second y-axis for the accuracy
    ax2.set_ylabel('Accuracy', color='tab:red')  # Add color to the accuracy axis
    ax2.set_ylim([0, 1])
    lns4 = ax2.plot(x, train_acc, 'b--', marker='s', label='Train Accuracy')  # Use dashed lines and square markers
    lns5 = ax2.plot(x, val_acc, 'g--', marker='s', label='Validation Accuracy')
    lns6 = ax2.plot(x, test_acc, 'r--', marker='s', label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Combine legends from both axes
    lns = lns1 + lns2 + lns3 + lns4 + lns5 + lns6
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, shadow=True)  # Move legend below x-axis

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the legend below the x-axis
    plt.title(header)

    # Create the directory if it does not exist and save the plot and data
    directory = './diagrams'
    os.makedirs(directory, exist_ok=True)
    plt.savefig(os.path.join(directory, fig_name))
    np.savez(os.path.join(directory, fig_name.replace('.png', '.npz')), 
             train_loss=train_loss, val_loss=val_loss, train_acc=train_acc, val_acc=val_acc, test_acc=test_acc)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_corpus_torch(args, device):
    """
    Loads input corpus from gcn/data directory, torch tensor version

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :returns All data input files loaded (as well the training/test data).
    Returns: 
        adj: sequential graph
        adj1: semantic graph
        adj2: syntactic graph
    """

    adjs = []
    for adj in ['seq','sem','syn']:
        logger.info("Loading {} graph".format(adj))
        if args.run_id is not None:
            try:
                adjs.append(pkl.load(open('./saved_graphs/run_{}/{}.{}_adj'.format(args.run_id,args.dataset,adj),'rb')))
                logger.info("Successfully loaded {} graph".format(adj))
            except Exception as e:
                logger.info('Unable to locate run_{}/{}.{}_adj in the directory'.format(args.run_id,args.dataset,adj))
        else: 
            adjs.append(pkl.load(open('./data/{}.{}_adj'.format(args.dataset,adj),'rb')))
    
    data = json.load(open('./data/{}_data.json'.format(args.dataset),'r'))
    train_ids, test_ids, corpus, labels, vocab, word_id_map, id_word_map, label_list = data

    num_labels = len(label_list)
    train_size = len(train_ids)

    val_size = int(0.1*len(train_ids))
    test_size = len(test_ids)

    labels = np.asarray(labels[:train_size]+[0]*len(vocab)+labels[train_size:])
    print(len(labels))


    idx_train = range(train_size-val_size)
    idx_val = range(train_size-val_size, train_size)
    idx_test = range(train_size+len(vocab), train_size+len(vocab)+test_size)
    
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_test[test_mask] = labels[test_mask]

    

    # seq, sem, syn = adjs[0], adjs[1], adjs[2]
    adjs_new = []
    for adj in adjs:
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adjs_new.append(adj)
    # seq = seq + seq.T.multiply(seq.T > seq) - seq.multiply(seq.T > seq)
    # sem = sem + sem.T.multiply(sem.T > sem) - sem.multiply(sem.T > sem)
    # syn = syn + syn.T.multiply(syn.T > syn) - syn.multiply(syn.T > syn)

    # tensor
    # adj = torch.sparse_csr_tensor(adj.indptr, adj.indices, adj.data, dtype=torch.float).to_sparse_coo().to(device)
    # adj1 = torch.sparse_csr_tensor(adj1.indptr, adj1.indices, adj1.data, dtype=torch.float).to_sparse_coo().to(device)
    # adj2 = torch.sparse_csr_tensor(adj2.indptr, adj2.indices, adj2.data, dtype=torch.float).to_sparse_coo().to(device)
    # features = torch.sparse_csr_tensor(features.indptr, features.indices, features.data, dtype=torch.float).to_sparse_coo().to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    train_mask = torch.tensor(train_mask, dtype=torch.float).to(device)
    val_mask = torch.tensor(val_mask, dtype=torch.float).to(device)
    test_mask = torch.tensor(test_mask, dtype=torch.float).to(device)

    return adjs_new, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, val_size,test_size, num_labels
    
def get_edge_tensor_list(adj_list, device):
    """
    
    Args:
        adj_list [list]: list of adjencies
    """
    indice_list, data_list = [], []
    for adj in adj_list:
        row = torch.tensor(adj.row, dtype=torch.long).to(device)
        col = torch.tensor(adj.col, dtype=torch.long).to(device)
        data = torch.tensor(adj.data, dtype=torch.float).to(device)
        indice = torch.stack((row,col),dim=0)
        indice_list.append(indice)
        data_list.append(data)
    return indice_list, data_list

def get_edge_tensor(adj):
    row = torch.tensor(adj.row, dtype=torch.long)
    col = torch.tensor(adj.col, dtype=torch.long)
    data = torch.tensor(adj.data, dtype=torch.float)
    indice = torch.stack((row,col),dim=0)
    return indice, data

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

def preprocess_features_origin(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def preprocess_adj_mix(adj):
    adj_normalized = adj + sp.eye(adj.shape[0])
    return sparse_to_tuple(adj)

def preprocess_adj_tensor(adj, device):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return torch.sparse_coo_tensor(np.stack([adj_normalized.row, adj_normalized.col], axis=0), adj_normalized.data, adj_normalized.shape, dtype=torch.float).to(device)

def preprocess_adj_mix_tensor(adj, device):
    adj_normalized = adj + sp.eye(adj.shape[0])
    # return torch.sparse_csr_tensor(crow_indices=adj.indptr, col_indices=adj.indices, values=adj.data, dtype=torch.float).to_sparse_coo().to(device)
    return torch.tensor(adj.todense(), dtype=torch.float).to(device)

def pickle_graph(graph_type:str, dataset, graph_adj, graph_saved_path):
    """
    Function to pickle graph using context manager

    Args:
        graph_type [str]: name of graph to be serialised
        dataset [str]: name of dataset
    """
    logger = logging.getLogger(__name__)

    start_time = time.time()
    logger.info(f"Persisting {graph_type} graph")

    with open(os.path.join(graph_saved_path, '{}.{}_adj').format(dataset, graph_type[:3]), 'wb') as f:
        pkl.dump(graph_adj, f)
    logger.info("Successfully persisted {} graph. Serialisation took {} seconds".format(graph_type, round(time.time()-start_time),2))

def set_torch_seed (seed:int=148):
    """
    Function to randomly generate seed for torch to ensure reproducability 

    Args:
        seed [int]: seed number
    """
    seed=148
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def setup_logging(log_path, log_name, timestamp, log_filename='model', max_bytes=1048576, backup_count=3):
    """
    Set up logging with RotatingFileHandler.

    Args:
    - log_path (str): Path to the directory where logs will be stored.
    - log_filename (str): Base name of the log file (timestamp will be appended).
    - max_bytes (int): Maximum size of each log file before it is rotated.
    - backup_count (int): Number of backup log files to keep.
    """
    # Create the log directory if it doesn't exist
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Set up logging
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    # Create a RotatingFileHandler
    log_file = os.path.join(log_path, f"{log_filename}_{timestamp}.log")
    handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    # Create a StreamHandler to print logs to the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Adjust log level if needed
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def pickle_artifact(args, filename, timestamp, artifact):

    """
    Function to pickle file 

    Args:
        args [args | class of args]: arguments
        filename [str]: name of file to be pickled
        timestamp [timestamp]: datetime of run_<timestamp> to store pickled file
        artifact: any data artifact to be persisted

    Returns:
        None
    """
    # make directory if folder does not exist
    if not os.path.exists(os.path.join(args.save_path,'run_{}'.format(timestamp))):
        logger.info(os.path.join(args.save_path,'run_{}'.format(timestamp)))
        os.makedirs(os.path.join(args.save_path,'run_{}'.format(timestamp)))

    # pickle file to directory
    try:
        with open(os.path.join(args.save_path ,'run_{}/{}.pkl'.format(timestamp, filename)), 'wb') as f:
            pkl.dump(artifact, f)
    except Exception as e:
        logger.info("There was an issue in pickling the file. Check the inputs again")
