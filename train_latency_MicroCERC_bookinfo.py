#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 20:33:33 2022

@author: ruyuexin
"""

import time
from utils_microcerc import *
from typing import List
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
idx = args.indx
atype = args.atype


class Simple:
    def __init__(self, global_now_time, global_end_time, label, root_cause, dir):
        self.global_now_time = global_now_time
        self.global_end_time = global_end_time
        self.label = label
        self.root_cause = root_cause
        self.dir = dir


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
    # pps = ['p50', 'p90', 'p99']
    pps = ['p50']
    simples: List[Simple] = [
        Simple(
            1706308740, 1706309520, 'label-productpage-net-latency-1', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-net-1'
        ),
        Simple(
            1706309640, 1706310420, 'label-productpage-net-latency-2', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-net-2'
        ),
        Simple(
            1706310540, 1706311320, 'label-productpage-net-latency-3', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-net-3'
        ),
        Simple(
            1706311440, 1706312220, 'label-productpage-net-latency-4', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-net-4'
        ),
        Simple(
            1706312340, 1706313120, 'label-productpage-net-latency-5', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-net-5'
        ),
        Simple(
            1706286240, 1706287020, 'label-productpage-cpu-1', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-cpu-1'
        ),
        Simple(
            1706287140, 1706287920, 'label-productpage-cpu-2', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-cpu-2'
        ),
        Simple(
            1706288040, 1706288820, 'label-productpage-cpu-3', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-cpu-3'
        ),
        Simple(
            1706288940, 1706289720, 'label-productpage-cpu-4', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-cpu-4'
        ),
        Simple(
            1706289840, 1706290620, 'label-productpage-cpu-5', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-cpu-5'
        ),
        Simple(
            1706290740, 1706291520, 'label-productpage-mem-1', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-1'
        ),
        Simple(
            1706291640, 1706292420, 'label-productpage-mem-2', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-2'
        ),
        Simple(
            1706292540, 1706293320, 'label-productpage-mem-3', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-3'
        ),
        Simple(
            1706293440, 1706294220, 'label-productpage-mem-4', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-4'
        ),
        Simple(
            1706294340, 1706295120, 'label-productpage-mem-5', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-5'
        ),
        Simple(
            1706295240, 1706296020, 'label-productpage-mem-6', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-6'
        ),
        Simple(
            1706296140, 1706296920, 'label-productpage-mem-7', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-7'
        ),
        Simple(
            1706297040, 1706297820, 'label-productpage-mem-8', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-8'
        ),
        Simple(
            1706297940, 1706298720, 'label-productpage-mem-9', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-9'
        ),
        Simple(
            1706298840, 1706299620, 'label-productpage-mem-10', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-10'
        ),
        Simple(
            1706299740, 1706300520, 'label-productpage-mem-11', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-11'
        ),
        Simple(
            1706300640, 1706301420, 'label-productpage-mem-12', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-12'
        ),
        Simple(
            1706301540, 1706302320, 'label-productpage-mem-13', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-13'
        ),
        Simple(
            1706302440, 1706303220, 'label-productpage-mem-14', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-14'
        ),
        Simple(
            1706303340, 1706304120, 'label-productpage-mem-15', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-15'
        ),
        Simple(
            1706304240, 1706305020, 'label-productpage-mem-16', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-16'
        ),
        Simple(
            1706305140, 1706305920, 'label-productpage-mem-17', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-17'
        ),
        Simple(
            1706306040, 1706306820, 'label-productpage-mem-18', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-18'
        ),
        Simple(
            1706306940, 1706307720, 'label-productpage-mem-19', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-19'
        ),
        Simple(
            1706307840, 1706308620, 'label-productpage-mem-20', 'productpage',
            'abnormal/bookinfo/productpage/bookinfo-productpage-mem-20'
        ),
    ]
    namespaces = ['bookinfo', 'hipster', 'cloud-sock-shop', 'horsecoder-test']
    for pp in pps:
        for simple in simples:
            print(simple.label)
            all_data = pd.DataFrame()
            for namespace in namespaces:
                all_data_ns = pd.read_csv(
                    '/Users/zhuyuhan/Documents/391-WHU/experiment/researchProject/MicroCERC/data/' + simple.dir + '/' + namespace + '/latency.csv')
                all_data_ns = df_time_limit_normalization(all_data_ns, simple.global_now_time, simple.global_end_time)
                if all_data.empty:
                    all_data = all_data_ns
                else:
                    all_data = pd.merge(all_data, all_data_ns, on='timestamp', how='outer')
            name = [i for i in all_data.columns if i != 'timestamp' and pp in i]
            data = all_data[name]

            data_sample_size = data.shape[0]
            data_variable_size = data.shape[1]
            # torch.manual_seed(CONFIG.seed)
            # if CONFIG.cuda:
            #     torch.cuda.manual_seed(CONFIG.seed)

            # ================================================
            # get data: experiments = {synthetic SEM, ALARM}
            # ================================================
            train_data = data

            # ===================================
            # load modules
            # ===================================
            # Generate off-diagonal interaction graph
            off_diag = np.ones([data_variable_size, data_variable_size]) - np.eye(data_variable_size)

            # add adjacency matrix A
            num_nodes = data_variable_size
            adj_A = np.zeros((num_nodes, num_nodes))

            if CONFIG.encoder == 'mlp':
                encoder = MLPEncoder(data_variable_size * CONFIG.x_dims, CONFIG.x_dims, CONFIG.encoder_hidden,
                                     int(CONFIG.z_dims), adj_A,
                                     batch_size=CONFIG.batch_size,
                                     do_prob=CONFIG.encoder_dropout, factor=CONFIG.factor).double()
            elif CONFIG.encoder == 'sem':
                encoder = SEMEncoder(data_variable_size * CONFIG.x_dims, CONFIG.encoder_hidden,
                                     int(CONFIG.z_dims), adj_A,
                                     batch_size=CONFIG.batch_size,
                                     do_prob=CONFIG.encoder_dropout, factor=CONFIG.factor).double()

            if CONFIG.decoder == 'mlp':
                decoder = MLPDecoder(data_variable_size * CONFIG.x_dims,
                                     CONFIG.z_dims, CONFIG.x_dims, encoder,
                                     data_variable_size=data_variable_size,
                                     batch_size=CONFIG.batch_size,
                                     n_hid=CONFIG.decoder_hidden,
                                     do_prob=CONFIG.decoder_dropout).double()
            elif CONFIG.decoder == 'sem':
                decoder = SEMDecoder(data_variable_size * CONFIG.x_dims,
                                     CONFIG.z_dims, 2, encoder,
                                     data_variable_size=data_variable_size,
                                     batch_size=CONFIG.batch_size,
                                     n_hid=CONFIG.decoder_hidden,
                                     do_prob=CONFIG.decoder_dropout).double()

            # ===================================
            # set up training parameters
            # ===================================
            if CONFIG.optimizer == 'Adam':
                optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=CONFIG.lr)
            elif CONFIG.optimizer == 'LBFGS':
                optimizer = optim.LBFGS(list(encoder.parameters()) + list(decoder.parameters()),
                                        lr=CONFIG.lr)
            elif CONFIG.optimizer == 'SGD':
                optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()),
                                      lr=CONFIG.lr)

            scheduler = lr_scheduler.StepLR(optimizer, step_size=CONFIG.lr_decay,
                                            gamma=CONFIG.gamma)

            # Linear indices of an upper triangular mx, used for acc calculation
            triu_indices = get_triu_offdiag_indices(data_variable_size)
            tril_indices = get_tril_offdiag_indices(data_variable_size)

            if CONFIG.prior:
                prior = np.array([0.91, 0.03, 0.03, 0.03])  # hard coded for now
                print("Using prior")
                print(prior)
                log_prior = torch.DoubleTensor(np.log(prior))
                log_prior = torch.unsqueeze(log_prior, 0)
                log_prior = torch.unsqueeze(log_prior, 0)
                log_prior = Variable(log_prior)

                if CONFIG.cuda:
                    log_prior = log_prior.cuda()

            if CONFIG.cuda:
                encoder.cuda()
                decoder.cuda()
                triu_indices = triu_indices.cuda()
                tril_indices = tril_indices.cuda()
            gamma = args.gamma
            eta = args.eta

            t_total = time.time()
            best_ELBO_loss = np.inf
            best_NLL_loss = np.inf
            best_MSE_loss = np.inf
            best_epoch = 0
            best_ELBO_graph = []
            best_NLL_graph = []
            best_MSE_graph = []
            # optimizer step on hyparameters
            c_A = CONFIG.c_A
            lambda_A = CONFIG.lambda_A
            h_A_new = torch.tensor(1.)
            h_tol = CONFIG.h_tol
            k_max_iter = int(CONFIG.k_max_iter)
            h_A_old = np.inf

            E_loss = []
            N_loss = []
            M_loss = []
            start_time = time.time()
            try:
                for step_k in range(k_max_iter):
                    # print(step_k)
                    while c_A < 1e+20:
                        for epoch in range(CONFIG.epochs):
                            # print(epoch)
                            ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = train(epoch, best_ELBO_loss, lambda_A, c_A,
                                                                                   optimizer)
                            E_loss.append(ELBO_loss)
                            N_loss.append(NLL_loss)
                            M_loss.append(MSE_loss)
                            if ELBO_loss < best_ELBO_loss:
                                best_ELBO_loss = ELBO_loss
                                best_epoch = epoch
                                best_ELBO_graph = graph

                            if NLL_loss < best_NLL_loss:
                                best_NLL_loss = NLL_loss
                                best_epoch = epoch
                                best_NLL_graph = graph

                            if MSE_loss < best_MSE_loss:
                                best_MSE_loss = MSE_loss
                                best_epoch = epoch
                                best_MSE_graph = graph

                        # print("Optimization Finished!")
                        # print("Best Epoch: {:04d}".format(best_epoch))
                        if ELBO_loss > 2 * best_ELBO_loss:
                            break

                        # update parameters
                        A_new = origin_A.data.clone()
                        h_A_new = _h_A(A_new, data_variable_size)
                        if h_A_new.item() > gamma * h_A_old:
                            c_A *= eta
                        else:
                            break

                    # update parameters
                    # h_A, adj_A are computed in loss anyway, so no need to store
                    h_A_old = h_A_new.item()
                    lambda_A += c_A * h_A_new.item()

                    if h_A_new.item() <= h_tol:
                        break

                # print("Steps: {:04d}".format(step_k))
                # print("Best Epoch: {:04d}".format(best_epoch))

                # test()
                # print (best_ELBO_graph)
                # print(best_NLL_graph)
                # print (best_MSE_graph)

                graph = origin_A.data.clone().cpu().numpy()
                graph[np.abs(graph) < 0.1] = 0
                graph[np.abs(graph) < 0.2] = 0
                graph[np.abs(graph) < 0.3] = 0

            except KeyboardInterrupt:
                print('Done!')

            end_time = time.time()
            # print("Time spent: ",end_time-start_time)
            adj = graph
            if not np.any(adj):
                print(simple.label + ' is absent')
                continue
            org_G = nx.from_numpy_matrix(adj, parallel_edges=True, create_using=nx.DiGraph)
            # pos = nx.circular_layout(org_G)
            # nx.draw(org_G, pos=pos, with_labels=True)
            # plt.savefig("metrics_causality.png")

            # PageRank in networkx
            # G = nx.from_numpy_matrix(adj.T, parallel_edges=True, create_using=nx.DiGraph)
            # scores = nx.pagerank(G, max_iter=1000)
            # print(sorted(scores.items(), key=lambda item:item[1], reverse=True))

            # PageRank
            from sknetwork.ranking import PageRank

            pagerank = PageRank()
            scores = pagerank.fit_transform(np.abs(adj.T))

            score_dict = {}
            for i, s in enumerate(scores):
                score_dict[name[i]] = s
            sorted_scores = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
            count = 0
            with open('MicroCERC/bookinfo/service/' + simple.label + '-' + pp + '.log', "a") as output_file:
                print('root cause: ' + simple.root_cause, file=output_file)
                for sorted_score in sorted_scores:
                    count += 1
                    print(sorted_score, file=output_file)
                    if ('edge' in simple.root_cause and simple.root_cause in sorted_score[0]) or (
                            'edge' not in simple.root_cause and simple.root_cause in sorted_score[0] and 'edge' not in
                            sorted_score[0]):
                        print("topK: " + str(count), file=output_file)
                        break
