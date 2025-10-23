import sys
import glob
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
# import torch_sparse
import pickle as pkl
import random
import networkx as nx
from normalization import fetch_normalization, row_normalize, aug_normalized_adjacency
from time import perf_counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn


from sklearn import metrics

def std_scaler(dfs):
    scaler = StandardScaler()
    scaler.fit(pd.concat(dfs))
    # print('scaler', dfs[0].columns)
    print('scaler--------------------------', scaler.mean_, scaler.scale_)
    # for df in dfs:
    #     print('df distribution', pd.concat([df.mean(axis='index'), df.std(axis='index')], axis=1))
    return scaler

def mm_scaler(dfs):
    scaler = MinMaxScaler()
    scaler.fit(pd.concat(dfs[:-1]))
    print('scaler', dfs[0].columns)
    print('scaler', scaler.min_, scaler.scale_)
    for df in dfs:
        print('df distribution', pd.concat([df.min(axis='index'), df.max(axis='index')], axis=1))
    return scaler

def rob_scaler(dfs):
    scaler = RobustScaler(quantile_range=(0, 100))
    scaler.fit(pd.concat(dfs[:-1]))
    # print('scaler', dfs[0].columns)
    # print('scaler', scaler.mean_, scaler.scale_)
    # for df in dfs:
    #     print('df distribution', pd.concat([df.mean(axis='index'), df.std(axis='index')], axis=1))
    return scaler

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

def batched_block_diagonal(sparse_coo):
    """

    @param sparse_coo: [bs, h, w]
    @return: sparse coo [bs*h, bs*w]
    """
    nnz = sparse_coo._nnz()
    shape = sparse_coo.size()
    indices = sparse_coo.coalesce().indices()
    # [b, h, w] -> [b*h, b*w]
    new_shape = (shape[0] * shape[1], shape[0] * shape[1])

    new_indices = torch.empty(2, nnz, device=indices.device, dtype=indices.dtype)
    # indices: [b,h,w] -> [h+b*H, w+b*W]
    new_indices[0, :] = indices[1, :] + indices[0, :] * shape[1]
    new_indices[1, :] = indices[2, :] + indices[0, :] * shape[2]

    val = sparse_coo.coalesce().values()
    return torch.sparse.FloatTensor(indices=new_indices, values=val, size=new_shape)

def preprocess_csm(dfs, in_len, seq_len):
    """
    Segment time series by sliding window -- 2*10 seconds for history, 60*10 seconds for prediction
    TODO: train on n-1 days?
    :param dfs: list of pandas dataframes
    """
    features, diffs = [], []
    for df in dfs:
        # Calculate 30-day Simple Moving Average (SMA)
        df.insert(loc=0, column='thruster_force_input', value=df['Thruster Force'].rolling(10, min_periods=1).mean())
        df.insert(loc=0, column='in_flow_rate_input', value=df['In Flow Rate'].rolling(10, min_periods=1).mean())
        df.insert(loc=0, column='inclination_input', value=df['Inclination'].rolling(10, min_periods=1).mean())
        # df.loc[:, 'env'] = np.random.normal(size=df.shape[0])
        pad = pd.DataFrame(np.zeros_like(df[:in_len]), index=df.index[:in_len], columns=df.columns)
        df = pd.concat([pad, df])
        df_diff = (df - df.shift(1)).map(lambda x: x**(1.0/3) if x > 0 else -(-x)**(1.0/3))
        df_diff.iloc[0] = 0

        features.append(df)
        diffs.append(df_diff)
    names = features[0].columns.tolist()#[:-2]
    scaler = std_scaler(dfs)
    scaler_diff = std_scaler(diffs)
    for d in range(len(dfs)):
        features[d] = torch.Tensor(scaler.transform(features[d]))
        diffs[d] = torch.Tensor(scaler_diff.transform(diffs[d]))

    return features, diffs, scaler, scaler_diff, names

def load_CSM(dataset_str, in_len, seq_len, normalization="AugNormAdj", fold_id=0,
             split_ratio=0.1, BATCH_SIZE=64, nts=False,
             cuda=True, mask_head=False, need_loader=True):
    """
    Load Citation Networks Datasets.
    """
    data = []
    for file in glob.glob('**', recursive=True):
        if file.endswith("_out__no_lead_zero_in_flow.csv"):
            df = pd.read_csv(file, header=0)
            df = df[['Inclination', 'In Flow Rate', 'Thruster Force',
                     'Weight on Bit', 'Torque on Bit', 'Drilling Speed', 'diff_Distance', 'diff_Depth',
                     'Internal Pressure', 'Annular Pressure', 'Out Flow Rate', 'Distance', 'Depth']] # 'Fluid Loss'
            data.append(df)
            # print(file, df.min(), df.max())

    # three-folds validation by three days
    if fold_id == 0:
        features, diffs, scaler, scaler_diff, names = preprocess_csm(data, in_len=in_len, seq_len=seq_len)
    elif fold_id == 1:
        features, diffs, scaler, scaler_diff, names = preprocess_csm([data[1], data[2], data[0]], in_len=in_len, seq_len=seq_len)
    elif fold_id == 2:
        features, diffs, scaler, scaler_diff, names = preprocess_csm([data[2], data[0], data[1]], in_len=in_len, seq_len=seq_len)
    else:
        raise ValueError('Invalid fold_id')

    adj = np.zeros((14, 14))

    # adj[0, :] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # other factors ~ N(0, 1)
    adj[0, :] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # inclination input
    adj[1, :] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # inflow rate input
    adj[2, :] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # thruster force (Load) input
    adj[3, :] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # inclination
    adj[4, :] = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # inflow rate
    adj[5, :] = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # thruster force (Load)
    adj[6, :] = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # weight on bit
    adj[7, :] = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # torque on bit
    adj[8, :] = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0])  # drilling speed
    adj[9, :] = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0])  # distance
    adj[10, :] = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # depth
    adj[11, :] = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # internal pressure
    adj[12, :] = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0])  # annular pressure
    adj[13, :] = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0])  # fluid loss

    if not nts:
        adj += np.eye(14)
    else:
        adj = adj.T
        d = adj.shape[0]
        lag_adj = np.concatenate([adj, np.eye(d)], axis=1)
        ins_adj = np.concatenate([np.zeros((d,d)), adj], axis=1)
        adj = np.concatenate([lag_adj, ins_adj])
        print(f'A with {adj.shape[0]/14} lag', adj.shape)
    # porting to pytorch
    train_x = features[:-1]
    test_x = features[-1:]

    train_y = diffs[:-1]
    test_y = diffs[-1:]
    # train_y = torch.Tensor(diffs[idx_train])
    # test_y = torch.Tensor(diffs[idx_test])

    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    return adj, (train_x, train_y), (test_x, test_y), (scaler, scaler_diff), names

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def Evaluation(output, labels):
    preds = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    '''
    binary_pred = preds
    binary_pred[binary_pred > 0.0] = 1
    binary_pred[binary_pred <= 0.0] = 0
    '''
    num_correct = 0
    binary_pred = np.zeros(preds.shape).astype('int')
    for i in range(preds.shape[0]):
        k = labels[i].sum().astype('int')
        topk_idx = preds[i].argsort()[-k:]
        binary_pred[i][topk_idx] = 1
        for pos in list(labels[i].nonzero()[0]):
            if labels[i][pos] and labels[i][pos] == binary_pred[i][pos]:
                num_correct += 1

    print('total number of correct is: {}'.format(num_correct))
    #print('preds max is: {0} and min is: {1}'.format(preds.max(),preds.min()))
    #'''
    return metrics.f1_score(labels, binary_pred, average="micro"), metrics.f1_score(labels, binary_pred, average="macro")



def sparse_mx_to_torch_sparse_tensor(sparse_mx, device=None):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    tensor = torch.sparse.FloatTensor(indices, values, shape)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def get_spectral_rad(sparse_tensor, tol=1e-5):
    """Compute spectral radius from a tensor"""
    A = sparse_tensor.data.coalesce().cpu()
    A_scipy = sp.coo_matrix((np.abs(A.values().numpy()), A.indices().numpy()), shape=A.shape)
    return np.abs(sp.linalg.eigs(A_scipy, k=1, return_eigenvectors=False)[0]) + tol

def projection_norm_inf(A, kappa=0.99, transpose=False):
    """ project onto ||A||_inf <= kappa return updated A"""
    # TODO: speed up if needed
    v = kappa
    if transpose:
        A_np = A.T.clone().detach().cpu().numpy()
    else:
        A_np = A.clone().detach().cpu().numpy()
    x = np.abs(A_np).sum(axis=-1)
    for idx in np.where(x > v)[0]:
        # read the vector
        a_orig = A_np[idx, :]
        a_sign = np.sign(a_orig)
        a_abs = np.abs(a_orig)
        a = np.sort(a_abs)

        s = np.sum(a) - v
        l = float(len(a))
        for i in range(len(a)):
            # proposal: alpha <= a[i]
            if s / l > a[i]:
                s -= a[i]
                l -= 1
            else:
                break
        alpha = s / l
        a = a_sign * np.maximum(a_abs - alpha, 0)
        # verify
        assert np.isclose(np.abs(a).sum(), v, atol=1e-4)
        # write back
        A_np[idx, :] = a
    A.data.copy_(torch.tensor(A_np.T if transpose else A_np, dtype=A.dtype, device=A.device))
    return A

def projection_norm_inf_and_1(A, kappa_inf=0.99, kappa_1=None, inf_first=True):
    """ project onto ||A||_inf <= kappa return updated A"""
    # TODO: speed up if needed
    v_inf = kappa_inf
    v_1 = kappa_inf if kappa_1 is None else kappa_1
    A_np = A.clone().detach().cpu().numpy()
    if inf_first:
        A_np = projection_inf_np(A_np, v_inf)
        A_np = projection_inf_np(A_np.T, v_1).T
    else:
        A_np = projection_inf_np(A_np.T, v_1).T
        A_np = projection_inf_np(A_np, v_inf)
    A.data.copy_(torch.tensor(A_np, dtype=A.dtype, device=A.device))
    return A

def projection_inf_np(A_np, v):
    x = np.abs(A_np).sum(axis=-1)
    for idx in np.where(x > v)[0]:
        # read the vector
        a_orig = A_np[idx, :]
        a_sign = np.sign(a_orig)
        a_abs = np.abs(a_orig)
        a = np.sort(a_abs)

        s = np.sum(a) - v
        l = float(len(a))
        for i in range(len(a)):
            # proposal: alpha <= a[i]
            if s / l > a[i]:
                s -= a[i]
                l -= 1
            else:
                break
        alpha = s / l
        a = a_sign * np.maximum(a_abs - alpha, 0)
        # verify
        assert np.isclose(np.abs(a).sum(), v, atol=1e-6)
        # write back
        A_np[idx, :] = a
    return A_np

def clip_gradient(model, clip_norm=10):
    """ clip gradients of each parameter by norm """
    for param in model.parameters():
        torch.nn.utils.clip_grad_norm(param, clip_norm)
    return model

def l_1_penalty(model, alpha=0.1):
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += alpha * torch.sum(torch.abs(param))
    return regularization_loss

class AdditionalLayer(torch.nn.Module):
    def __init__(self, model, num_input, num_output, activation=torch.nn.ReLU()):
        super().__init__()
        self.model = model
        self.add_module("model", self.model)
        self.activation = activation
        if isinstance(activation, torch.nn.Module):
            self.add_module("activation", self.activation)
        self.func = torch.nn.Linear(num_input, num_output, bias=False)

    def forward(self, *input):
        x = self.model(*input)
        x = self.activation(x)
        return self.func(x)

def load_raw_graph(dataset_str = "amazon-all"):
    txt_file = 'data/' + dataset_str + '/adj_list.txt'
    graph = {}
    with open(txt_file, 'r') as f:
        cur_idx = 0
        for row in f:
            row = row.strip().split()
            adjs = []
            for j in range(1, len(row)):
                adjs.append(int(row[j]))
            graph[cur_idx] = adjs
            cur_idx += 1
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    normalization="AugNormAdj"
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    return adj

def load_txt_data(dataset_str = "amazon-all", portion = '0.06'):
    adj = load_raw_graph(dataset_str)
    idx_train = list(np.loadtxt('data/' + dataset_str + '/train_idx-' + str(portion) + '.txt', dtype=int))
    idx_val = list(np.loadtxt('data/' + dataset_str + '/test_idx.txt', dtype=int))
    idx_test = list(np.loadtxt('data/' + dataset_str + '/test_idx.txt', dtype=int))
    labels = np.loadtxt('data/' + dataset_str + '/label.txt')
    with open('data/' + dataset_str + '/meta.txt', 'r') as f:
        num_nodes, num_class = [int(w) for w in f.readline().strip().split()]

    features = sp.identity(num_nodes)
    
    # porting to pytorch
    features = sparse_mx_to_torch_sparse_tensor(features).float()
    labels = torch.FloatTensor(labels)
    #labels = torch.max(labels, dim=1)[1]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, num_nodes, num_class

# def sgc_precompute(features, adj, degree):
#     t = perf_counter()
#     adj_index = adj.coalesce().indices()
#     adj_value = adj.coalesce().values()
#     features_index = features.coalesce().indices()
#     features_value = features.coalesce().values()
#     m = adj.shape[0]
#     n = adj.shape[1]
#     k = features.shape[1]
#
#     for i in range(degree):
#         #features = torch.spmm(adj, features)
#         features_index, features_value = torch_sparse.spspmm(adj_index, adj_value, features_index, features_value, m, n, k)
#     precompute_time = perf_counter()-t
#     return torch.sparse.FloatTensor(features_index, features_value, torch.Size(features.shape)), precompute_time

class SparseDropout(torch.nn.Module):
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob=1-dprob

    def forward(self, x, training):
        if training:
            mask=((torch.rand(x._values().size())+(self.kprob)).floor()).type(torch.bool)
            rc=x._indices()[:,mask]
            val=x._values()[mask]*(1.0/self.kprob)
            return torch.sparse.FloatTensor(rc, val, torch.Size(x.shape))
        else:
            return x

def compute_metrics(ys, preds, scaler):
    """
    Compute prediction losses of CSM
    """
    # ys, preds = ys.squeeze(-1), preds.squeeze(-1) # S, T, N
    mse = np.mean((ys - preds)**2, axis=0)
    mae = np.mean(np.abs(ys - preds), axis=0)
    mape = np.mean(np.abs(ys - preds)/(np.abs(ys) + 1e-6), axis=0)
    r2 = 1 - np.sum((ys - preds)**2, axis=0)/(np.sum((ys - ys.mean(axis=0)) ** 2, axis=0) + 1e-6)
    return scaler**2 * mse, scaler * mae, mape, r2

def squeeze_diag(x):
    out = []
    for i in range(x.shape[-1]):
        out.append(x[..., i, i])
    out = torch.stack(out, dim=-1)
    return out

def init_network(modules):
    for m in modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Parameter):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

def param2attn(mask, param, n_time, n_nodes, n_heads, hdim):
    markoff_value = -1e6
    param = param.squeeze()
    param = param.reshape(n_time, n_nodes, n_nodes, n_heads)
    param = param * mask + markoff_value * (1 - mask)
    attention = torch.softmax((param / (hdim ** (1 / 2))).clamp(-5, 5), dim=1).float()
    return attention

def add_noise(data):
    add_mean = (torch.rand_like(data) - 0.5) * 4 # uniform range -2 ~ 2
    add_std = (torch.rand_like(data) + 0.2) * 2 # uniform range 0.2 ~ 2
    return (data + add_mean) * add_std

def arrange_input(data, label, context):
    '''
    Arrange a single time series into overlapping short sequences.

    Args:
      data: time series of shape (T, dim).
      context: length of short sequences.
    '''
    assert context >= 1 and isinstance(context, int)
    # input = torch.zeros(len(data) - context, context, data.shape[1],
    #                     dtype=torch.float32, device='cuda')
    # target = torch.zeros(len(label) - context, context, label.shape[1],
    #                      dtype=torch.float32, device='cuda')
    # for i in range(context):
    #     start = i
    #     end = len(data) - context + i
    #     input[:, i, :] = data[start:end]
    #     target[:, i, :] = label[start:end]
    input, target = [], []
    for i in range(0, len(data), context):
        d = data[i:i + context]

        input.append(d)
        target.append(label[i:i + context])
    while input[-1].shape[0] < context:
        input, target = input[:-1], target[:-1]
    input, target = torch.stack(input), torch.stack(target)
    return input.detach(), target.detach()
