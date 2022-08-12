#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)
    len_sents = [len(s) for s in sents]
    max_len = max(len_sents)
    for i in range(len(sents)):
        padded = sents[i] + [pad_token for _ in range(max_len - len_sents[i])]
        sents_padded.append(padded)
    # --- One line version ---
    # sents_padded = [s+[pad_token for _ in range(max_len-len(s))] for s in sents]

    ### END YOUR CODE

    return sents_padded



def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data



def batch_iter(data, batch_size, shuffle=False):
    """
    Yield batches of source and target sentences reverse sorted by source length (largest to smallest).
    # [JC] it's sorted by source length, then correspondingly re-order the target sentence
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True) # [JC] sorted by source sentence length
        src_sents = [e[0] for e in examples] # [JC]reorder by sorted length
        tgt_sents = [e[1] for e in examples] # [JC] reorder correspondingly

        yield src_sents, tgt_sents


if __name__ == '__main__':
    sents = ["i eat meat".split(), "i got a bbe daa".split(), "nothing".split()]
    print(sents)
    token = "<PAD>"
    result = pad_sents(sents,token)
    print(result)