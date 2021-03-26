
import numpy as np
import argparse
import json
import sys
import os
import tick
import pickle
import random
from collections import defaultdict
import math

# External libraries
from tick.hawkes.simulation import SimuHawkesExpKernels
from tick.hawkes.inference import HawkesADM4, HawkesSumGaussians


def learn_adm4(events, end_time, return_learner=False, verbose=False, **kwargs):
    learner_mle = HawkesADM4(**kwargs, verbose=verbose)
    learner_mle.fit(events, end_time)
    if return_learner:
        return learner_mle
    return learner_mle.baseline, learner_mle.adjacency, learner_mle.score()


def create_item_list(data):
    item_dict = defaultdict(int)
    item_dict2 = defaultdict(int)
    events = []

    cnt = 0
    for index, (item, time) in enumerate(data):
        if item_dict.get(item):
            events[item_dict.get(item)].append(time)
        else:
            events.append([time])
            item_dict[item] = cnt
            item_dict2[cnt] = item
            cnt+=1

    events = [np.array(x) for x in events]

    return events, item_dict, item_dict2


def superpose_user_list(user_list, user, users):
    sp_user_list = user_list[user][:]
    for u in users:
        sp_user_list.extend(user_list[u])

    sp_user_list = sorted(sp_user_list, key=lambda x:x[1])

    return sp_user_list


def intensity(mu, A, user_list, t, w):
    user_list = [event for event in user_list if event[1]<=t]
    lamb = mu

    for c in range(len(lamb)):
        for event in user_list:
            lamb[c] += w*math.exp(-w*(t-event[1])) * A[c][event[0]]

    lamb = [max(lamb_c, 0) for lamb_c in lamb]

    return lamb


def supintensity(mu, A, user_list, t, w, tstep=10, M=10):
    if len(user_list)==0:
        return sum(mu)

    user_list = [event for event in user_list if event[1] <= t]

    mt = sum(mu)
    MT = [mt for m in range(M)]
    for m in range(M):
        t_current = t + 1.0 * m * tstep / M

        for c in range(len(mu)):
            for event in user_list:
                MT[m] += math.exp(-w * (t_current - event[1])) * A[c][event[0]]

    mt = max(0,max(MT))

    return mt


def prediction(mu, A, user_list, t_start, t_end, w, num_gen_events=10):

    if t_start == None:
        t_start = user_list[-1][1]

    all_events = user_list[:]
    gen_events = []

    t = t_start

    while t<t_end and len(gen_events)<num_gen_events:

        mt = supintensity(mu, A, all_events, t, w)

        s = np.random.exponential(1.0/mt, size=1)[0]
        U = np.random.rand()

        lamb_ts = intensity(mu, A, all_events, t, w)
        mts = sum(lamb_ts)

        if t+s>t_end or U>mts/mt:
            t += s
        else:
            d = np.random.choice(len(lamb_ts), size=1, p=np.divide(lamb_ts, mts))[0]

            t += s
            gen_events.append((d, t))
            all_events.append((d, t))

    return gen_events


def result_update(train_user_list, test_user_list, item_dict, gen_events, top_n):
    train_events = list(map(lambda x:x[0], train_user_list))

    test_events = list(map(lambda x:x[0], test_user_list))

    if item_dict is not None:
        train_events = list(map(lambda x: item_dict.get(x, -1), train_events))
        test_events = list(map(lambda x: item_dict.get(x, -1), test_events))

    gen_events = list(map(lambda x:x[0], gen_events))

    for index, event in enumerate(gen_events):
        if event in train_events:
            gen_events.pop(index)

    gen_events = gen_events[:top_n]

    return gen_events, test_events




def metric(gen_events, test_events):
    real = set(test_events)
    gen = set(gen_events)

    if len(real) == 0:
        return 1.0, 1.0, 1.0
    if len(gen) == 0:
        return 0, 0, 0

    precision = 1.0 * len(real & gen) / len(gen)
    recall = 1.0 * len(real & gen) / len(real)
    f1 = 1.0 * 2 * precision * recall / (precision + recall + 1e-5)

    return precision, recall, f1


def dimension_dist(seq1, seq2):
    seq1 = [d for d in seq1 if len(d)>0]
    seq2 = [d for d in seq2 if len(d)>0]

    dist = np.zeros((len(seq1), len(seq2)))

    for index1, s1 in enumerate(seq1):
        for index2, s2 in enumerate(seq2):
            T = max(np.max(s1), np.max(s2))

            if len(s1)>len(s2):
                s2 = np.concatenate((s2, T * np.ones((len(s1)-len(s2), ))))
                dist[index1, index2] = np.sum(np.abs(s1 - s2))
            elif len(s2)>len(s1):
                s1 = np.concatenate((s1, T * np.ones((len(s2)-len(s1), ))))
                dist[index1, index2] = np.sum(np.abs(s1 - s2))
            else:
                dist[index1, index2] = np.sum(np.abs(s1 - s2))

    return dist


def sequence_dist(seq1, seq2, beta=10000, J=1):
    D = dimension_dist(seq1, seq2)
    m, n = D.shape

    p = 1.0 / m * np.ones((m, ))
    q = 1.0 / n * np.ones((n, ))
    # T = np.dot(p, q.T)
    a = p
    b = q
    C = np.exp(-1.0/beta*D)

    for j in range(J):
        b = q / np.dot(C.T, a)
        a = p / np.dot(C, b)

    T = np.dot(np.dot(np.diag(a), C), np.diag(b))

    return np.sum(D * T)


def merge_adj(sub_adj, all_adj, item_dict):
    m, n = sub_adj.shape

    for i in range(m):
        real_i = item_dict.get(i)
        for j in range(n):
            real_j = item_dict.get(j)
            all_adj[real_i, real_j] += sub_adj[i, j]

    return all_adj


def merge_baseline(sub_user_baseline, all_user_baseline, item_dict):
    for index, baseline in enumerate(sub_user_baseline):
        real_i = item_dict.get(index)
        all_user_baseline[real_i] += baseline

    return all_user_baseline


def main(args):
    np.set_printoptions(threshold=np.inf)

    data_path = '../../Data/'
    with open(data_path + args.dataset + '.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']

    with open(args.param_filename, 'r') as param_file:
        param_dict = json.load(param_file)
    print('Complete loading data.')

    num_sp = param_dict['num_sp']
    inference_param_dict = param_dict['adm4']
    decay = inference_param_dict['decay']
    num_gen_events = param_dict['num_gen_events']
    top_n = param_dict['top_n']
    num_iter_bandit = param_dict['num_iter_bandit']
    rate_bandit = param_dict['rate_bandit']

    if param_dict['seq_dist_done'] is False:
        time_seqs = []
        all_seq_dist = []
        for user_list in train_user_list:
            seq, item_dict, item_dict2 = create_item_list(user_list)
            time_seqs.append(seq)

        for i in range(user_size):
            all_seq_dist.append([])
            for j in range(user_size):
                if i == j:
                    all_seq_dist[i].append(1e5)
                else:
                    all_seq_dist[i].append(sequence_dist(time_seqs[i], time_seqs[j]))

        all_seq_dist = [np.array(seq_dist) for seq_dist in all_seq_dist]
        with open(data_path + args.dataset + '_seq_dist.pickle', 'wb') as f:
            pickle.dump(all_seq_dist, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(data_path + args.dataset + '_seq_dist.pickle', 'rb') as f:
            all_seq_dist = pickle.load(f)

    print('Complete computing distance between sequences.')


    all_baseline = np.zeros((user_size, item_size))
    all_adj = np.zeros((item_size, item_size))
    all_sp_user_list = []

    for sp_user in range(user_size):
        sel_count = np.zeros((user_size,))
        reward = max(all_seq_dist[sp_user]) - all_seq_dist[sp_user]

        for iter in range(num_iter_bandit):
            cur_choice = np.random.choice(user_size, num_sp, p=reward/sum(reward))
            sel_count[cur_choice] += 1

            sp_user_list = superpose_user_list(train_user_list, sp_user, cur_choice)
            trainset, item_dict, item_dict2 = create_item_list(sp_user_list)
            baseline_adm4, adj_adm4, likelihood = learn_adm4(trainset, None, **inference_param_dict)

            reward[cur_choice] += rate_bandit * likelihood

            print('iter: %d, likelihood: %.3f'%(iter, likelihood))
            print('current choice:')
            print(cur_choice)


        print('Complete bandit problem phase.')

        last_choice = np.argsort(reward)[-num_sp:]
        sp_user_list = superpose_user_list(train_user_list, sp_user, last_choice)
        trainset, item_dict, item_dict2 = create_item_list(sp_user_list)
        sub_baseline, sub_adj, likelihood = learn_adm4(trainset, None, **inference_param_dict)

        all_baseline[sp_user] = merge_baseline(sub_baseline, all_baseline[sp_user], item_dict2)
        all_adj = merge_adj(sub_adj, all_adj, item_dict2)
        all_sp_user_list.append(sp_user_list)

        print('Complete user %d \'s Hawkes process learning.'%(sp_user))



    all_precision = 0
    all_recall = 0
    all_f1 = 0
    for user in range(user_size):

        gen_events = prediction(mu=all_baseline[user],
                                A=all_adj,
                                user_list=all_sp_user_list[user],
                                t_start=None,
                                t_end=200000,
                                w=decay,
                                num_gen_events=num_gen_events)

        print('trainset:')
        print(all_sp_user_list[user])
        print('testset:')
        print(gen_events)

        gen_events, test_events = result_update(train_user_list=train_user_list[user],
                                                test_user_list=test_user_list[user],
                                                item_dict=None,
                                                gen_events=gen_events,
                                                top_n=top_n)
        # print(len(gen_events), len(test_events))

        precision, recall, f1 = metric(gen_events, test_events)
        print('user: %d, precision: %.3f%%, recall: %.3f%%, f1: %.3f%%'%(user, 100.0*precision, 100.0*recall, 100.0*f1))

        all_precision += precision
        all_recall += recall
        all_f1 += f1

        print('all_precision: %.3f%%, all_recall: %.3f%%, all_f1: %.3f%%' % (
                100.0 * all_precision / (user+1), 100.0 * all_recall / (user+1), 100.0 * all_f1 / (user+1)))

    print('Complete!')
    print('all_precision: %.3f%%, all_recall: %.3f%%, all_f1: %.3f%%' % (
        100.0 * all_precision / user_size, 100.0 * all_recall / user_size, 100.0 * all_f1 / user_size))


def main2(args):
    np.set_printoptions(threshold=np.inf)

    data_path = '../../Data/'
    with open(data_path + args.dataset + '.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']

    with open(args.param_filename, 'r') as param_file:
        param_dict = json.load(param_file)
    print('Complete loading data.')

    num_sp = param_dict['num_sp']
    inference_param_dict = param_dict['adm4']
    decay = inference_param_dict['decay']
    num_gen_events = param_dict['num_gen_events']
    top_n = param_dict['top_n']
    num_iter_bandit = param_dict['num_iter_bandit']
    rate_bandit = param_dict['rate_bandit']

    all_user_list = [item_time  for user_list in train_user_list for item_time in user_list]
    all_user_list = sorted(all_user_list, key=lambda x:x[1])

    trainset = [[] for i in range(item_size * 2)]
    for item, time in all_user_list:
        trainset[item].append(time)
    for index, time_list in enumerate(trainset):
        trainset[index] = np.array(time_list)

    print(trainset)
    print('user_size:')
    print(user_size)
    print('item_size:')
    print(item_size)

    if param_dict['use_last_adj'] is True:
        with open(data_path + args.dataset + '_all_adj.pickle', 'rb') as f:
            adj = pickle.load(f)
    else:
        baseline, adj, likelihood = learn_adm4(trainset, None, **inference_param_dict)
        # print(adj)
        with open(data_path + args.dataset + '_all_adj.pickle', 'wb') as f:
            pickle.dump(adj, f, protocol=pickle.HIGHEST_PROTOCOL)


    all_precision = 0
    all_recall = 0
    all_f1 = 0
    for user in range(user_size):


        intensity_list = np.zeros((item_size * 2,))
        # user_list = list(map(lambda x:(item_dict.get(x[0], -1), x[1]), train_user_list[user]))
        user_list = train_user_list[user]
        train_item_list = list(map(lambda x:x[0], train_user_list[user]))

        for (item, time) in user_list:
            intensity_list += adj[:, item] * np.exp(decay * time)

        recommendation = intensity_list.argsort()[::-1]
        gen_events = []
        for item in recommendation:
            # item -= item_size if item >= item_size else 0
            if item not in train_item_list and item < item_size:
                gen_events.append(item)
            if len(gen_events)>= top_n:
                break

        # test_events = list(map(lambda x:item_dict.get(x[0], -1), test_user_list[user]))
        test_events = list(map(lambda x:x[0] - (item_size if x[0] >= item_size else 0), test_user_list[user]))

        precision, recall , f1 = metric(gen_events, test_events)
        print('user: %d'%(user))
        print('purchasing history:')
        print(user_list)
        print('prediction:')
        print(gen_events)
        print('test:')
        print(test_events)
        print('precision: %.3f%%, recall: %.3f%%, f1: %.3f%%' % (
        100.0 * precision, 100.0 * recall, 100.0 * f1))

        all_precision += precision
        all_recall += recall
        all_f1 += f1

        print('all_precision: %.3f%%, all_recall: %.3f%%, all_f1: %.3f%%' % (
            100.0 * all_precision / (user + 1), 100.0 * all_recall / (user + 1), 100.0 * all_f1 / (user + 1)))





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        dest='dataset',
                        type=str,
                        help="Name of dataset to be preprocessed")
    parser.add_argument('--params',
                        dest='param_filename',
                        type=str,
                        default='params.json',
                        help="Input parameter file (JSON)")
    args = parser.parse_args()
    # main(args)

    # Compared to SLIM
    main2(args)