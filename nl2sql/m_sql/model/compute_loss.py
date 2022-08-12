import random

import torch
import torch.nn.functional as F
from sql_utils.utils_tableqa import get_wnop
from .nl2sql_layer import Seq2SQL_v1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Loss_s2s(score, g_pnt_idxs):
    """
    score = [B, T, max_seq_length]
    """
    #         WHERE string part
    loss = 0

    for b, g_pnt_idxs1 in enumerate(g_pnt_idxs):
        ed = len(g_pnt_idxs1) - 1
        score_part = score[b, :ed]
        loss += F.cross_entropy(score_part, torch.tensor(g_pnt_idxs1[1:]).to(device))  # +1 shift.
    return loss


def Loss_sw_se(s_sn, s_sc, s_sa, s_wnop, s_wc, s_wo, s_tags, s_match,
               g_sn, g_sc, g_sa, g_wnop, g_wc, g_wo, g_tags, g_value_match):
    """
    :param s_wv: score  [ B, n_conds, T, score]
    :param g_wn: [ B ]
    :param g_wvi: [B, conds, pnt], e.g. [[[0, 6, 7, 8, 15], [0, 1, 2, 3, 4, 15]], [[0, 1, 2, 3, 16], [0, 7, 8, 9, 16]]]
    :return:
    """
    g_wn, _ = get_wnop(g_wnop)
    loss_sn = F.binary_cross_entropy(s_sn, (torch.tensor(g_sn)-1.0).to(device), reduction='sum')
    loss_sc = Loss_col(s_sc, g_sc)
    loss_sa = Loss_agg_wo(s_sa, g_sn, g_sa)
    loss_wnop = Loss_wnop(s_wnop, g_wnop)
    loss_wc = Loss_col(s_wc, g_wc)
    loss_wo = Loss_agg_wo(s_wo, g_wn, g_wo)

    # tag loss & match loss
    loss_tag = Loss_tags(s_tags, g_tags, neg_samp=True)
    loss_match = Loss_match(s_match, g_value_match)
    # losses = ['loss_sn', 'loss_sc', 'loss_sa', 'loss_wnop', 'loss_wc', 'loss_wo', 'loss_tag', 'loss_match']
    loss = loss_sn + loss_sc + loss_sa + loss_wnop + loss_wc + loss_wo + loss_tag + loss_match
    return loss


def Loss_wnop(s, g):
    loss = F.cross_entropy(s, torch.tensor(g).to(device), reduction='sum')
    return loss


def Loss_match(s_match, g_match):
    loss = 0
    bS, col_n, val_n = s_match.shape
    for b, g_match1 in enumerate(g_match):
        col_num = min(len(g_match1), col_n)
        g_match1_flatten = []
        for g_match11 in g_match1:
            g_match1_flatten += g_match11
        val_num = min(max(g_match1_flatten) + 1, val_n)
        g_match_tensor = [[0.0 for _ in range(val_num)] for _ in range(col_num)]
        for col, vals in enumerate(g_match1):
            for v in vals:
                if col < col_num and v < val_num:
                    g_match_tensor[col][v] = 1.0
        loss += F.binary_cross_entropy(s_match[b, :col_num, :val_num],
                                       torch.tensor(g_match_tensor).to(device))

    return loss

def Loss_tags(s_tags, g_tags, neg_samp=False):
    g_tags_tensor = Seq2SQL_v1.pad_tensor(g_tags).float()
    loss = 0
    for b, tags in enumerate(g_tags):
        if neg_samp:
            positives = []
            negatives = []
            for i, t in enumerate(tags):
                if t == 1:
                    positives.append(i)
                else:
                    negatives.append(i)
            pos_len = len(positives)
            negatives = random.sample(negatives, min(pos_len, len(negatives)))
            indexes = torch.tensor(positives + negatives, device=device)
            loss += F.binary_cross_entropy(s_tags[b, indexes], g_tags_tensor[b, indexes])
        else:
            length = len(tags)
            loss += F.binary_cross_entropy(s_tags[b, :length], g_tags_tensor[b, :length])
    return loss


def Loss_col(s_wc, g_wc):
    # Construct index matrix
    bS, max_h_len = s_wc.shape
    im = torch.zeros([bS, max_h_len]).to(device)
    for b, g_wc1 in enumerate(g_wc):
        for g_wc11 in g_wc1:
            if g_wc11 < max_h_len:
                im[b, g_wc11] = 1.0
    # Construct prob.
    loss = F.binary_cross_entropy(s_wc, im, reduction='sum') / max_h_len
    return loss


def Loss_agg_wo(s_wo, g_wn, g_wo):

    # Construct index matrix
    loss = 0
    for b, g_wn1 in enumerate(g_wn):
        g_wn1 = min(g_wn1, s_wo.size(1))
        if g_wn1 == 0:
            continue
        g_wo1 = g_wo[b]
        s_wo1 = s_wo[b]
        loss += F.cross_entropy(s_wo1[:g_wn1], torch.tensor(g_wo1[:g_wn1]).to(device))

    return loss

