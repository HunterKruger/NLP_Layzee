import jieba
import gensim
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
from Levenshtein import distance as l_dist
from jaro import jaro_winkler_metric as jaro_sim
from nltk.translate.bleu_score import sentence_bleu
from config import config


def get_counts(row):
    s1, s2 = row['seg_sent1'], row['seg_sent2']
    set_s1 = set(s1)
    set_s2 = set(s2)
    return len(set_s1.intersection(set_s2)), len(set_s1-set_s2), len(set_s2-set_s1) 


def get_ngram_bleu(row):
    s1, s2 = row[config.SENTENCE1_FIELD], row[config.SENTENCE2_FIELD]
    list_s1 = [x for x in s1]
    list_s2 = [x for x in s2]
    b1 = sentence_bleu(list_s1, list_s2, weights=(1, 0, 0, 0))
    b2 = sentence_bleu(list_s1, list_s2, weights=(0.5, 0.5, 0, 0))
    b3 = sentence_bleu(list_s1, list_s2, weights=(0.33, 0.33, 0.33, 0))
    b4 = sentence_bleu(list_s1, list_s2, weights=(0.25, 0.25, 0.25, 0.25))
    return b1, b2, b3, b4


def get_distances(row):
    """
    L2 Distance
    L1 Distance
    Cosine Similarity
    Canberra Distance
    Pearson Correlation 
    # Jaccard Similarity

    """
    sent_emd_l1 = row['emb_sent1']
    sent_emd_l2 = row['emb_sent2']

    return pairwise_distances([sent_emd_l1],[sent_emd_l2],metric='euclidean')[0][0],\
           pairwise_distances([sent_emd_l1],[sent_emd_l2],metric='manhattan')[0][0],\
           pairwise_distances([sent_emd_l1],[sent_emd_l2],metric='cosine')[0][0], \
           distance.canberra(sent_emd_l1,sent_emd_l2),\
           np.corrcoef(np.array(sent_emd_l1),  np.array(sent_emd_l2))[0][1]

def jaccard(row):
    s1, s2 = row[config.SENTENCE1_FIELD], row[config.SENTENCE2_FIELD]
    set_s1 = set([x for x in s1])
    set_s2 = set([x for x in s2])
    return float(len(set_s1.intersection(set_s2)) / len(set_s1.union(set_s2)))

def segmentation(x, stopwords=None):
    seg_list = jieba.cut(x)
    if stopwords is None:
        result = [x for x in seg_list]
    else:
        result = [x for x in seg_list if x not in stopwords]
    return  result


def get_text_embedding(x, word2vec):
    token_id_list = [word2vec.key_to_index[x] for x in x if x in word2vec.key_to_index.keys()]   # get token ids
    ## calculate embedding
    emb = np.zeros(len(word2vec[0]))
    for idx in token_id_list:
        emb += word2vec[idx]
    emb /= len(x)
    return emb

def feature_engineering(df, s1, s2):
    '''
    df: DataFrame
    s1: column name of sentence 1 
    s2: column name of sentence 2
    '''
    df_ = df.copy()

    if config.STOPWORDS_FILE is None:
        stopwords_list = None
    else: 
        stopwords_list = []
        f = open(config.STOPWORDS_FILE)
        lines = f.read()
        stopwords_list = lines.split('\n')

    df_['seg_sent1'] = df_[s1].apply(lambda x: segmentation(x,stopwords_list))
    df_['seg_sent2'] = df_[s2].apply(lambda x: segmentation(x,stopwords_list))

    df_['len_sent1'] = df_[s1].apply(lambda x: len(x))
    df_['len_sent2'] = df_[s2].apply(lambda x: len(x))

    df_['len_diff'] = df_['len_sent1'] - df_['len_sent2']
    df_['len_diff'] = df_['len_diff'].apply(lambda x: abs(x))

    df_['len_diff_normalized'] = df_['len_diff']/(df_['len_sent1']+df_['len_sent2'])

    df_['len_seg_sent1'] = df_[s1].apply(lambda x: len(x))
    df_['len_seg_sent2'] = df_[s2].apply(lambda x: len(x))

    df_['len_seg_diff'] = df_['len_seg_sent1'] - df_['len_seg_sent2']
    df_['len_seg_diff'] = df_['len_seg_diff'].apply(lambda x: abs(x))
    df_['len_seg_diff_normalized'] = df_['len_seg_diff']/(df_['len_seg_sent1']+df_['len_seg_sent2'])

    df_['jaro_similarity'] = df_.apply(lambda x: jaro_sim(x[s1], x[s2]),axis=1)

    df_['jaccard'] = df_.apply(jaccard, axis=1, result_type='expand')

    df_['edit_distance'] = df_.apply(lambda x: l_dist(x[s1],x[s2]),axis=1)
    df_['edit_distance_normalized'] = df_['edit_distance']/(df_['len_sent1']+df_['len_sent2'])

    df_[['common_wc','s1_uq_wc','s2_uq_wc']] = df_.apply(get_counts, axis=1,result_type='expand')

    df_['common_wc_nzd'] = df_['common_wc'] / (df_['len_seg_sent1'] + df_['len_seg_sent2'])
    df_['s1_uq_wc_nzd'] = df_['s1_uq_wc'] / (df_['len_seg_sent1'])
    df_['s2_uq_wc_nzd'] = df_['s2_uq_wc'] / (df_['len_seg_sent2'])

    df_[['bleu1','bleu2','bleu3','bleu4']] = df_.apply(get_ngram_bleu, axis=1, result_type='expand')

    word2vec = gensim.models.KeyedVectors.load_word2vec_format(config.WORD2VEC_FILE, binary=False)

    df_['emb_sent1'] = df_[s1].apply(lambda x: get_text_embedding(x, word2vec))
    df_['emb_sent2'] = df_[s2].apply(lambda x: get_text_embedding(x, word2vec))

    df_[['l2_dist','l1_dist','cos_dist','cbr_dist','pearson_corr']] = df_.apply(get_distances, axis=1,result_type='expand')

    df_.drop(['seg_sent1','seg_sent2','emb_sent1','emb_sent2'], inplace=True, axis=1)

    return df_
