# from unicodedata import normalize
import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from time import perf_counter
from sklearn.preprocessing import normalize

# AORM stuffs
from aorm import *
from tqdm import tqdm
from scipy.sparse import csr_matrix


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    # print(f'# Loaded objects: {objects}')

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    print(f'# x: {x.size}, tx: {tx.size}, allx: {allx.size}')
    print(f'# y: {y.size}, ty: {ty.size}, ally: {ally.size}')

    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # print(f'# NetworkX adj: {type(adj)}')
    # print(adj)
    dense_adj = np.array(adj.todense().A)
    # print(f'# Dense adj: {dense_adj}')
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # dense_adj = adj.todense()
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    adj, features = preprocess_citation(adj, features, normalization)

    # print(f"# Adjacency matrix type: {adj}")

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test, dense_adj

def sgc_precompute(features, adj, degree):
    t = perf_counter()
    # print(f"# Adjacency matrix: {adj}")
    # print(f"# Feature matrix: {features}")
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter()-t
    return features, precompute_time

def aorm_precompute(features, s_adj, degree, sp_only=False):
    t = perf_counter()

    h_aorm = s_adj.copy()

    for k_aorm, power in smx_aorm_iterator(s_adj, k=degree, shortest_only=sp_only, method = 'sp_mm'):
        weight = 1.0/(power + 2)                                    # simple weight calculation
        h_aorm = h_aorm + k_aorm * weight                           # apply simple weighting
        normed_aorm = normalize(k_aorm, axis=1, norm='l1')          # simple matrix normalization
        adj_mx = sp.csr_matrix(normed_aorm)                         # convert to sparse coo matrix
        adj_mx_tensor = sparse_mx_to_torch_sparse_tensor(adj_mx)    # convert to pytorch sparse coo tensor   
        features = torch.spmm(adj_mx_tensor, features)              # sparse coo matrix multiplication

    precompute_time = perf_counter()-t
    return features, precompute_time

# def aorm_precompute(features, dense_adj, degree, sp_only=False):
#     t = perf_counter()

#     # np.fill_diagonal(dense_adj, 0)
#     # n, e = len(dense_adj), np.count_nonzero(dense_adj)
#     # mu_degree = round(2*e / n, 2)
#     # print(f'# |V|: {n}, |E|: {e}, mean degree: {mu_degree}')
#     # h_aorm = np.zeros_like(dense_adj)

#     sparse_adj = csr_matrix(dense_adj)
#     sparse_adj.setdiag(0)
#     sparse_adj.eliminate_zeros()
#     n, e = sparse_adj.shape[0], sparse_adj.nnz    
#     h_aorm = sparse_adj.copy()

#     # for k_aorm, power in tqdm(AormIterator(dense_adj, k=degree, shortest_only=sp_only, method = 'edge'), desc='# AORM'):
#     for k_aorm, power in tqdm(smx_aorm_iterator(sparse_adj, k=degree, shortest_only=sp_only, method = 'sp_mm'), desc='# AORM'):
#         weight = 1.0/(power + 2)                                    # simple weight calculation
#         h_aorm = h_aorm + k_aorm * weight                           # apply simple weighting
#         normed_aorm = normalize(k_aorm, axis=1, norm='l1')          # simple matrix normalization
#         adj_mx = sp.csr_matrix(normed_aorm)                         # convert to sparse coo matrix
#         adj_mx_tensor = sparse_mx_to_torch_sparse_tensor(adj_mx)    # convert to pytorch sparse coo tensor   
#         features = torch.spmm(adj_mx_tensor, features)              # sparse coo matrix multiplication

#     precompute_time = perf_counter()-t
#     return features, precompute_time

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def load_reddit_data(data_path="data/", normalization="AugNormAdj", cuda=False):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    adj = adj + adj.T
    train_adj = adj[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    # print(f'# reddit adj: {type(adj)}')
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj))
    # dense_adj = np.array(adj.todense())    
    # print(f'# dense matrix: {dense_adj}')
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    train_adj = adj_normalizer(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    labels = torch.LongTensor(labels)
    if cuda:
        adj = adj.cuda()
        train_adj = train_adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
    return adj, train_adj, features, labels, train_index, val_index, test_index
