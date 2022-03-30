#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import numpy as np
import torch
import random
from sentence_transformers import util
from loguru import logger
from tools.file_io import load_pickle_file
from gensim.models.word2vec import Word2Vec


def setup_seed(seed):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def align_word_embedding(words_list_path, model_path, nhid):

    words_list = load_pickle_file(words_list_path)
    w2v_model = Word2Vec.load(model_path)
    ntoken = len(words_list)
    weights = np.zeros((ntoken, nhid))
    for i, word in enumerate(words_list):
        if word in w2v_model.wv.index_to_key:
            embedding = w2v_model.wv[word]
            weights[i] = embedding
    weights = torch.from_numpy(weights).float()
    return weights


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


# evaluation tools
def a2t(audio_embs, cap_embs, return_ranks=False):
    # audio to caption retrieval
    num_audios = int(audio_embs.shape[0] / 5)
    index_list = []

    ranks = np.zeros(num_audios)
    top1 = np.zeros(num_audios)
    for index in range(num_audios):
        # get query audio
        audio = audio_embs[5 * index].reshape(1, audio_embs.shape[1])

        # compute scores
        d = util.cos_sim(torch.Tensor(audio), torch.Tensor(cap_embs)).squeeze(0).numpy()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr


def t2a(audio_embs, cap_embs, return_ranks=False):
    # caption to audio retrieval
    num_audios = int(audio_embs.shape[0] / 5)

    audios = np.array([audio_embs[i]for i in range(0, audio_embs.shape[0], 5)])

    ranks = np.zeros(5 * num_audios)
    top1 = np.zeros(5 * num_audios)

    for index in range(num_audios):

        # get query captions
        queries = cap_embs[5 * index: 5 * index + 5]

        # compute scores
        d = util.cos_sim(torch.Tensor(queries), torch.Tensor(audios)).numpy()

        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr
