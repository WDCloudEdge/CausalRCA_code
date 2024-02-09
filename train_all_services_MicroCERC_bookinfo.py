#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 20:33:33 2022

@author: ruyuexin
"""

import time
from utils_microcerc import *
from typing import List
from train_latency_MicroCERC_bookinfo import Simple
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import *
from modules import *
from config import CONFIG
import warnings
import argparse

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--indx', type=int, default=0, help='index')
parser.add_argument('--atype', type=str, default='cpu-hog1_', help='anomaly type')
parser.add_argument('--gamma', type=float, default=0.25, help='gamma')
parser.add_argument('--eta', type=int, default=10, help='eta')
args = parser.parse_args()

CONFIG.cuda = torch.cuda.is_available()
CONFIG.factor = not CONFIG.no_factor


# torch.manual_seed(CONFIG.seed)
# if CONFIG.cuda:
#     torch.cuda.manual_seed(CONFIG.seed)


# compute constraint h(A) value
def _h_A(A, m):
    expm_A = matrix_poly(A * A, m)
    h_A = torch.trace(expm_A) - m
    return h_A


prox_plus = torch.nn.Threshold(0., 0.)


def stau(w, tau):
    w1 = prox_plus(torch.abs(w) - tau)
    return torch.sign(w) * w1


def update_optimizer(optimizer, original_lr, c_A):
    '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr


# ===================================
# training:
# ===================================
def train(epoch, best_val_loss, lambda_A, c_A, optimizer):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    shd_trian = []

    encoder.train()
    decoder.train()
    scheduler.step()

    # update optimizer
    optimizer, lr = update_optimizer(optimizer, CONFIG.lr, c_A)

    for i in range(1):
        data = train_data[i * data_sample_size:(i + 1) * data_sample_size]
        data = torch.tensor(data.to_numpy().reshape(data_sample_size, data_variable_size, 1))
        if CONFIG.cuda:
            data = data.cuda()
        data = Variable(data).double()

        optimizer.zero_grad()

        enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(
            data)  # logits is of size: [num_sims, z_dims]
        edges = logits
        # print(origin_A)
        dec_x, output, adj_A_tilt_decoder = decoder(data, edges, data_variable_size * CONFIG.x_dims, origin_A,
                                                    adj_A_tilt_encoder, Wa)

        if torch.sum(output != output):
            print('nan error\n')

        target = data
        preds = output
        variance = 0.

        # reconstruction accuracy loss
        loss_nll = nll_gaussian(preds, target, variance)

        # KL loss
        loss_kl = kl_gaussian_sem(logits)

        # ELBO loss:
        loss = loss_kl + loss_nll
        # add A loss
        one_adj_A = origin_A  # torch.mean(adj_A_tilt_decoder, dim =0)
        sparse_loss = CONFIG.tau_A * torch.sum(torch.abs(one_adj_A))

        # other loss term
        if CONFIG.use_A_connect_loss:
            connect_gap = A_connect_loss(one_adj_A, CONFIG.graph_threshold, z_gap)
            loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

        if CONFIG.use_A_positiver_loss:
            positive_gap = A_positive_loss(one_adj_A, z_positive)
            loss += .1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

        # compute h(A)
        h_A = _h_A(origin_A, data_variable_size)
        loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(
            origin_A * origin_A) + sparse_loss  # +  0.01 * torch.sum(variance * variance)

        # print(loss)
        loss.backward()
        loss = optimizer.step()

        myA.data = stau(myA.data, CONFIG.tau_A * lr)

        if torch.sum(origin_A != origin_A):
            print('nan error\n')

        # compute metrics
        graph = origin_A.data.clone().cpu().numpy()
        graph[np.abs(graph) < CONFIG.graph_threshold] = 0

        mse_train.append(F.mse_loss(preds, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())

    return np.mean(np.mean(kl_train) + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A


# ===================================
# main
# ===================================
if __name__ == '__main__':
    simples: List[Simple] = [
        Simple(
            1706322060, 1706322840, 'label-ratings-edge-net-latency-1', 'ratings-v1-edge',
            'abnormal/bookinfo/ratings-edge/bookinfo-ratings-edge-net-1'
        ),
        Simple(
            1706322960, 1706323740, 'label-ratings-edge-net-latency-2', 'ratings-v1-edge',
            'abnormal/bookinfo/ratings-edge/bookinfo-ratings-edge-net-2'
        ),
        Simple(
            1706323860, 1706324640, 'label-ratings-edge-net-latency-3', 'ratings-v1-edge',
            'abnormal/bookinfo/ratings-edge/bookinfo-ratings-edge-net-3'
        ),
        Simple(
            1706324760, 1706325540, 'label-ratings-edge-net-latency-4', 'ratings-v1-edge',
            'abnormal/bookinfo/ratings-edge/bookinfo-ratings-edge-net-4'
        ),
        Simple(
            1706325660, 1706326440, 'label-ratings-edge-net-latency-5', 'ratings-v1-edge',
            'abnormal/bookinfo/ratings-edge/bookinfo-ratings-edge-net-5'
        ),
        Simple(
            1706272740, 1706273520, 'label-ratings-edge-cpu-1', 'ratings-v1-edge',
            'abnormal/bookinfo/ratings-edge/bookinfo-ratings-edge-cpu-1'
        ),
        Simple(
            1706273640, 1706274420, 'label-ratings-edge-cpu-2', 'ratings-v1-edge',
            'abnormal/bookinfo/ratings-edge/bookinfo-ratings-edge-cpu-2'
        ),
        Simple(
            1706274540, 1706275320, 'label-ratings-edge-cpu-3', 'ratings-v1-edge',
            'abnormal/bookinfo/ratings-edge/bookinfo-ratings-edge-cpu-3'
        ),
        Simple(
            1706275440, 1706276220, 'label-ratings-edge-cpu-4', 'ratings-v1-edge',
            'abnormal/bookinfo/ratings-edge/bookinfo-ratings-edge-cpu-4'
        ),
        Simple(
            1706276340, 1706277120, 'label-ratings-edge-cpu-5', 'ratings-v1-edge',
            'abnormal/bookinfo/ratings-edge/bookinfo-ratings-edge-cpu-5'
        ),
        Simple(
            1706277240, 1706278020, 'label-ratings-edge-mem-1', 'ratings-v1-edge',
            'abnormal/bookinfo/ratings-edge/bookinfo-ratings-edge-mem-1'
        ),
        Simple(
            1706278140, 1706278920, 'label-ratings-edge-mem-2', 'ratings-v1-edge',
            'abnormal/bookinfo/ratings-edge/bookinfo-ratings-edge-mem-2'
        ),
        Simple(
            1706279040, 1706279820, 'label-ratings-edge-mem-3', 'ratings-v1-edge',
            'abnormal/bookinfo/ratings-edge/bookinfo-ratings-edge-mem-3'
        ),
        Simple(
            1706279940, 1706280720, 'label-ratings-edge-mem-4', 'ratings-v1-edge',
            'abnormal/bookinfo/ratings-edge/bookinfo-ratings-edge-mem-4'
        ),
        Simple(
            1706280840, 1706281620, 'label-ratings-edge-mem-5', 'ratings-v1-edge',
            'abnormal/bookinfo/ratings-edge/bookinfo-ratings-edge-mem-5'
        ),
    ]
    namespaces = ['bookinfo', 'hipster', 'cloud-sock-shop', 'horsecoder-test']
    for simple in simples:
        print(simple.label)
        all_data = pd.DataFrame()
        for namespace in namespaces:
            all_data_ns = pd.read_csv(
                '/Users/zhuyuhan/Documents/391-WHU/experiment/researchProject/MicroCERC/data/' + simple.dir + '/' + namespace + '/instance.csv')
            all_data_ns = df_time_limit_normalization(all_data_ns, simple.global_now_time, simple.global_end_time)
            if all_data.empty:
                all_data = all_data_ns
            else:
                all_data = pd.merge(all_data, all_data_ns, on='timestamp', how='outer')
        name = [i for i in all_data.columns if i != 'timestamp']
        data = all_data[name]
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import chisq

        cg = pc(data.to_numpy(), 0.05, chisq, False, 0, -1)
        adj = cg.G.graph

        # print('PC result')
        # print(adj)

        # Change the adj to graph
        G = nx.DiGraph()
        for i in range(len(adj)):
            for j in range(len(adj)):
                if adj[i, j] == -1:
                    G.add_edge(i, j)
                if adj[i, j] == 1:
                    G.add_edge(j, i)
        nodes = sorted(G.nodes())
        # print(nodes)
        adj = np.asarray(nx.to_numpy_matrix(G, nodelist=nodes))
        if not np.any(adj):
            print(simple.label + ' is absent')
            continue
        # org_G = nx.from_numpy_matrix(adj, parallel_edges=True, create_using=nx.DiGraph)
        # pos = nx.circular_layout(org_G)
        # nx.draw(org_G, pos=pos, with_labels=True)
        # plt.savefig("metrics_causality.png")

        # PageRank in networkx
        # G = nx.from_numpy_matrix(adj.T, parallel_edges=True, create_using=nx.DiGraph)
        # scores = nx.pagerank(G, max_iter=1000)

        # PageRank
        from sknetwork.ranking import PageRank

        pagerank = PageRank()
        scores = pagerank.fit_transform(np.abs(adj.T))

        score_dict = {}
        for i, s in enumerate(scores):
            score_dict[name[i]] = s
        sorted_scores = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
        count = 0
        with open('MicroCERC/bookinfo/instance/' + simple.label + '.log', "a") as output_file:
            print('root cause: ' + simple.root_cause, file=output_file)
            for sorted_score in sorted_scores:
                count += 1
                print(sorted_score, file=output_file)
                if ('edge' in simple.root_cause and simple.root_cause in sorted_score[0]) or (
                        'edge' not in simple.root_cause and simple.root_cause in sorted_score[0] and 'edge' not in
                        sorted_score[0]):
                    print("topK: " + str(count), file=output_file)
                    break
