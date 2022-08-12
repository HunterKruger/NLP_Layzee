import os, json
import random

import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def json_default_type_checker(o):
    """
    From https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python
    """
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def load_jsonl(path_file, toy_data=False, toy_size=4, shuffle=False, seed=1):
    data = []

    with open(path_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if toy_data and idx >= toy_size and (not shuffle):
                break
            t1 = json.loads(line.strip())
            data.append(t1)

    if shuffle and toy_data:
        # When shuffle required, get all the data, shuffle, and get the part of data.
        print(
            f"If the toy-data is used, the whole data loaded first and then shuffled before get the first {toy_size} data")

        random.Random(seed).shuffle(data)  # fixed
        data = data[:toy_size]

    return data


# Load data
def load_tableqa(path, toy_model, toy_size, no_hs_tok=False, aug=False):
    # Get data
    train_data, train_table = load_tableqa_data(path, mode='train', toy_model=toy_model, toy_size=toy_size, no_hs_tok=no_hs_tok, aug=aug)
    dev_data, dev_table = load_tableqa_data(path, mode='val', toy_model=toy_model, toy_size=toy_size, no_hs_tok=no_hs_tok)

    return train_data, train_table, dev_data, dev_table


def load_tableqa_data(path, mode='train', toy_model=False, toy_size=10, no_hs_tok=False, aug=False):
    """ Load training sets
    """
    if aug:
        mode = f"aug.{mode}"
        print('Augmented data is loaded!')

    path_sql = os.path.join(path, mode+'_tok.jsonl')
    if no_hs_tok:
        path_table = os.path.join(path, mode + '.tables.jsonl')
    else:
        path_table = os.path.join(path, mode+'_tok.tables.jsonl')

    data = []
    table = {}
    with open(path_sql) as f:
        for idx, line in enumerate(f):
            if toy_model and idx >= toy_size:
                break

            t1 = json.loads(line.strip())
            data.append(t1)

    with open(path_table) as f:
        for idx, line in enumerate(f):
            # if toy_model and idx > toy_size:
            #     break

            t1 = json.loads(line.strip())
            table[t1['id']] = t1

    return data, table


def get_loader(data_train, data_dev, bS, shuffle_train=True, shuffle_dev=False):
    train_loader = torch.utils.data.DataLoader(
        batch_size=bS,
        dataset=data_train,
        shuffle=shuffle_train,
        num_workers=4,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    dev_loader = torch.utils.data.DataLoader(
        batch_size=bS,
        dataset=data_dev,
        shuffle=shuffle_dev,
        num_workers=4,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    return train_loader, dev_loader


def get_fields_1(t1, tables, no_hs_t=False, no_sql_t=False):
    nlu1 = t1['question']
    nlu_t1 = t1['question_tok']
    tid1 = t1['table_id']
    sql_i1 = t1['sql']
    sql_q1 = ''
    col_val1 = t1['col_value']
    if no_sql_t:
        sql_t1 = None
    else:
        sql_t1 = t1['query_tok']

    tb1 = tables[tid1]
    if not no_hs_t:
        hs_t1 = tb1['header_tok']
    else:
        hs_t1 = []
    assert len(col_val1) == len(tb1['header'])
    hs1 = []
    for h, v in zip(tb1['header'], col_val1):
        if len(v) > 50:
            v = 'none'
        hs1.append(h + ',' + v)

    return nlu1, nlu_t1, tid1, sql_i1, sql_q1, sql_t1, tb1, hs_t1, hs1

def get_fields(t1s, tables, no_hs_t=False, no_sql_t=False):

    nlu, nlu_t, tid, sql_i, sql_q, sql_t, tb, hs_t, hs = [], [], [], [], [], [], [], [], []
    for t1 in t1s:
        if no_hs_t:
            nlu1, nlu_t1, tid1, sql_i1, sql_q1, sql_t1, tb1, hs_t1, hs1 = get_fields_1(t1, tables, no_hs_t, no_sql_t)
        else:
            nlu1, nlu_t1, tid1, sql_i1, sql_q1, sql_t1, tb1, hs_t1, hs1 = get_fields_1(t1, tables, no_hs_t, no_sql_t)

        nlu.append(nlu1)
        nlu_t.append(nlu_t1)
        tid.append(tid1)
        sql_i.append(sql_i1)
        sql_q.append(sql_q1)
        sql_t.append(sql_t1)

        tb.append(tb1)
        hs_t.append(hs_t1)
        hs.append(hs1)

    return nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hs


# Statistics
def get_wc1(conds):
    """
    [ [wc, wo, wv],
      [wc, wo, wv], ...
    ]
    """
    # wc1 = []
    # for cond in conds:
    #     wc1.append(cond[0])
    # return wc1
    wc1 = set()
    for cond in conds:
        wc1.add(cond[0])
    return sorted(list(wc1))


def get_wo1(conds, g_wc1):
    """
    [ [wc, wo, wv],
      [wc, wo, wv], ...
    ]
    """
    wo1 = [0 for _ in g_wc1]
    for cond in conds:
        c, o, _ = cond
        index = g_wc1.index(c)
        wo1[index] = o
    return wo1


def get_wv1(conds, g_wc1):
    """
    [ [wc, wo, wv],
      [wc, wo, wv], ...
    ]
    """
    wv1 = [[] for _ in g_wc1]
    for cond in conds:
        c, _, v = cond
        index = g_wc1.index(c)
        wv1[index].append(v)
    return wv1

wnop_map = {
    '0,1': 0, '1,1': 1, '1,2': 2, '1,3': 3, '2,1': 4, '2,2': 5, '2,3': 6
}

def get_g(sql_i):
    """ for backward compatibility, separated with get_g"""
    g_sn = []
    g_sc = []
    g_sa = []
    g_wnop = []
    g_wc = []
    g_wo = []
    g_wv = []
    for b, psql_i1 in enumerate(sql_i):
        cols = []
        aggs = []
        g_sn.append(len(psql_i1["sel"]))
        col_agg = sorted(list(zip(psql_i1["sel"], psql_i1["agg"])))
        for c, a in col_agg:
            cols.append(c)
            aggs.append(a)
        g_sc.append(cols)
        g_sa.append(aggs)

        conds = psql_i1['conds'][:3]
        wop = psql_i1['cond_conn_op']

        if len(psql_i1["agg"]) == len(psql_i1["sel"]):
            g_wc1 = get_wc1(conds)
            wn = len(g_wc1)
            g_wnop.append(wnop_map[str(wop) + ',' + str(wn)])
            g_wc.append(g_wc1)
            g_wo.append(get_wo1(conds, g_wc1))
            g_wv.append(get_wv1(conds, g_wc1))

        else:
            raise EnvironmentError

    return g_sn, g_sc, g_sa, g_wnop, g_wc, g_wo, g_wv


def get_g_wvi(t, g_wc):
    g_wvi = []
    g_tags = []
    g_value_match = []
    for b, t1 in enumerate(t):
        # g_wvi.append(t1['where_val_index'])
        g_tags.append(t1['value_tags'])
        vm = [[] for _ in g_wc[b]]
        wvi = [[] for _ in g_wc[b]]
        for i, (c, _, v) in enumerate(t1['sql']['conds'][:3]):
            index = g_wc[b].index(c)
            try:
                vm[index].append(t1['value_match'][i])
                wvi[index].append(t1['where_val_index'][i])
            except:
                vm[index].append(0)
                wvi[index].append([0, 1])
        g_value_match.append(vm)
        g_wvi.append(wvi)
    return g_wvi, g_tags, g_value_match


# BERT
def generate_inputs(tokenizer, nlu1_tok, hds1):
    tokens = []
    segment_ids = []

    tokens.append("[unused1]")  # [XLS]
    i_st_nlu = len(tokens)  # to use it later

    segment_ids.append(0)
    for token in nlu1_tok:
        tokens.append(token)
        segment_ids.append(0)
    i_ed_nlu = len(tokens)
    tokens.append("[SEP]")
    segment_ids.append(0)

    i_hds = []
    # for doc
    for i, hds11 in enumerate(hds1):
        i_st_hd = len(tokens)
        sub_tok = tokenizer.tokenize(hds11)
        tokens += sub_tok
        i_ed_hd = len(tokens)
        i_hds.append((i_st_hd, i_ed_hd))
        segment_ids += [1] * len(sub_tok)
        if i <= len(hds1)-1:
            tokens.append("[SEP]")
            segment_ids.append(1)
        else:
            raise EnvironmentError

    i_nlu = (i_st_nlu, i_ed_nlu)

    return tokens, segment_ids, i_nlu, i_hds

def gen_l_hpu(i_hds):
    """
    # Treat columns as if it is a batch of natural language utterance with batch-size = # of columns * # of batch_size
    i_hds = [(17, 18), (19, 21), (22, 23), (24, 25), (26, 29), (30, 34)])
    """
    l_hpu = []
    for i_hds1 in i_hds:
        for i_hds11 in i_hds1:
            l_hpu.append(i_hds11[1] - i_hds11[0])

    return l_hpu


def get_bert_output(model_bert, tokenizer, nlu_t, hds, max_seq_length):
    """
    Here, input is toknized further by WordPiece (WP) tokenizer and fed into BERT.

    INPUT
    :param model_bert:
    :param tokenizer: WordPiece toknizer
    :param nlu: Question
    :param nlu_t: CoreNLP tokenized nlu.
    :param hds: Headers
    :param hs_t: None or 1st-level tokenized headers
    :param max_seq_length: max input token length

    OUTPUT
    tokens: BERT input tokens
    nlu_tt: WP-tokenized input natural language questions
    orig_to_tok_index: map the index of 1st-level-token to the index of 2nd-level-token
    tok_to_orig_index: inverse map.

    """

    l_n = []
    l_hs = []  # The length of columns for each batch

    input_ids = []
    tokens = []
    segment_ids = []
    input_mask = []

    i_nlu = []  # index to retreive the position of contextual vector later.
    i_hds = []

    doc_tokens = []
    nlu_tt = []

    t_to_tt_idx = []
    tt_to_t_idx = []
    for b, nlu_t1 in enumerate(nlu_t):

        hds1 = hds[b]
        l_hs.append(len(hds1))


        # 1. 2nd tokenization using WordPiece
        tt_to_t_idx1 = []  # number indicates where sub-token belongs to in 1st-level-tokens (here, CoreNLP).
        t_to_tt_idx1 = []  # orig_to_tok_idx[i] = start index of i-th-1st-level-token in all_tokens.
        nlu_tt1 = []  # all_doc_tokens[ orig_to_tok_idx[i] ] returns first sub-token segement of i-th-1st-level-token
        for (i, token) in enumerate(nlu_t1):
            t_to_tt_idx1.append(
                len(nlu_tt1))  # all_doc_tokens[ indicate the start position of original 'white-space' tokens.
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tt_to_t_idx1.append(i)
                nlu_tt1.append(sub_token)  # all_doc_tokens are further tokenized using WordPiece tokenizer
        nlu_tt.append(nlu_tt1)
        tt_to_t_idx.append(tt_to_t_idx1)
        t_to_tt_idx.append(t_to_tt_idx1)

        l_n.append(len(nlu_tt1))
        #         hds1_all_tok = tokenize_hds1(tokenizer, hds1)



        # [CLS] nlu [SEP] col1 [SEP] col2 [SEP] ...col-n [SEP]
        # 2. Generate BERT inputs & indices.
        tokens1, segment_ids1, i_nlu1, i_hds1 = generate_inputs(tokenizer, nlu_tt1, hds1)
        input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)

        # Input masks
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask1 = [1] * len(input_ids1)

        # 3. Zero-pad up to the sequence length.
        while len(input_ids1) < max_seq_length:
            input_ids1.append(0)
            input_mask1.append(0)
            segment_ids1.append(0)

        assert len(input_ids1) == max_seq_length
        assert len(input_mask1) == max_seq_length
        assert len(segment_ids1) == max_seq_length

        input_ids.append(input_ids1)
        tokens.append(tokens1)
        segment_ids.append(segment_ids1)
        input_mask.append(input_mask1)

        i_nlu.append(i_nlu1)
        i_hds.append(i_hds1)

    # Convert to tensor
    all_input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(device)

    # 4. Generate BERT output.
    # all_encoder_layer, pooled_output = model_bert(all_input_ids, all_segment_ids, all_input_mask)
    sequence_output, pooled_output, all_encoder_layer = \
                                       model_bert(input_ids=all_input_ids,
                                                  token_type_ids=all_segment_ids,
                                                  attention_mask=all_input_mask,
                                                  output_hidden_states=True)
    all_encoder_layer = all_encoder_layer[1:]
    # 5. generate l_hpu from i_hds
    l_hpu = gen_l_hpu(i_hds)

    return all_encoder_layer, pooled_output, tokens, i_nlu, i_hds, \
           l_n, l_hpu, l_hs, \
           nlu_tt, t_to_tt_idx, tt_to_t_idx



def get_wemb_n(i_nlu, l_n, hS, num_hidden_layers, all_encoder_layer, num_out_layers_n):
    """
    Get the representation of each tokens.
    """
    bS = len(l_n)
    l_n_max = max(l_n)
    wemb_n = torch.zeros([bS, l_n_max, hS * num_out_layers_n]).to(device)
    for b in range(bS):
        # [B, max_len, dim]
        # Fill zero for non-exist part.
        l_n1 = l_n[b]
        i_nlu1 = i_nlu[b]
        for i_noln in range(num_out_layers_n):
            i_layer = num_hidden_layers - 1 - i_noln
            st = i_noln * hS
            ed = (i_noln + 1) * hS
            wemb_n[b, 0:(i_nlu1[1] - i_nlu1[0]), st:ed] = all_encoder_layer[i_layer][b, i_nlu1[0]:i_nlu1[1], :]
    return wemb_n


def get_wemb_h(i_hds, l_hpu, l_hs, hS, num_hidden_layers, all_encoder_layer, num_out_layers_h):
    """
    As if
    [ [table-1-col-1-tok1, t1-c1-t2, ...],
       [t1-c2-t1, t1-c2-t2, ...].
       ...
       [t2-c1-t1, ...,]
    ]
    """
    bS = len(l_hs)
    l_hpu_max = max(l_hpu)
    num_of_all_hds = sum(l_hs)
    wemb_h = torch.zeros([num_of_all_hds, l_hpu_max, hS * num_out_layers_h]).to(device)
    b_pu = -1
    for b, i_hds1 in enumerate(i_hds):
        for b1, i_hds11 in enumerate(i_hds1):
            b_pu += 1
            for i_nolh in range(num_out_layers_h):
                i_layer = num_hidden_layers - 1 - i_nolh
                st = i_nolh * hS
                ed = (i_nolh + 1) * hS
                wemb_h[b_pu, 0:(i_hds11[1] - i_hds11[0]), st:ed] \
                    = all_encoder_layer[i_layer][b, i_hds11[0]:i_hds11[1],:]


    return wemb_h



def get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length, num_out_layers_n=1, num_out_layers_h=1):

    bS = len(nlu_t)
    # get contextual output of all tokens from bert
    all_encoder_layer, pooled_output, tokens, i_nlu, i_hds,\
    l_n, l_hpu, l_hs, \
    nlu_tt, t_to_tt_idx, tt_to_t_idx = get_bert_output(model_bert, tokenizer, nlu_t, hds, max_seq_length)
    # all_encoder_layer: BERT outputs from all layers.
    # pooled_output: output of [CLS] vec.
    # tokens: BERT intput tokens
    # i_nlu: start and end indices of question in tokens
    # i_hds: start and end indices of headers


    # get the wemb
    wemb_n = get_wemb_n(i_nlu, l_n, bert_config.hidden_size, bert_config.num_hidden_layers, all_encoder_layer,
                        num_out_layers_n)

    wemb_h = get_wemb_h(i_hds, l_hpu, l_hs, bert_config.hidden_size, bert_config.num_hidden_layers, all_encoder_layer,
                        num_out_layers_h)
    wemb_cls = get_wemb_cls(bS, bert_config.hidden_size, all_encoder_layer, num_out_layers_n)

    return wemb_cls, wemb_n, wemb_h, l_n, l_hpu, l_hs, \
           nlu_tt, t_to_tt_idx, tt_to_t_idx


def get_wemb_cls(bS, hS, all_encoder_layer, num_out_layers_cls):
    wemb_cls = torch.zeros([bS, hS * num_out_layers_cls]).to(device)
    for batch_i in range(bS):
        for l_id in range(num_out_layers_cls):
            start = hS * l_id
            end = hS * (l_id + 1)
            wemb_cls[batch_i, start:end] = all_encoder_layer[-1 - l_id][batch_i, 0, :]
    return wemb_cls


def pred_sn(s_sn):
    pr_sn = []
    for s_sn1 in s_sn:
        pr_sn.append(2 if s_sn1.item() > 0.5 else 1)
    return pr_sn


def pred_sc(sn, s_sc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    ! Returned index is sorted!
    """
    # get g_num
    pr_sc = []
    for b, sn1 in enumerate(sn):
        s_sc1 = s_sc[b]
        pr_sc1 = np.argsort(-s_sc1.data.cpu().numpy())[:sn1]
        pr_sc1.sort()

        pr_sc.append(list(pr_sc1))
    return pr_sc


def pred_sa(sn, s_sa):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_sa = []
    for b, s_sa1 in enumerate(s_sa):
        n = sn[b]
        s_sa11 = []
        for i in range(n):
            s_sa11.append(s_sa1[i].argmax().item())
        pr_sa.append(s_sa11)
    return pr_sa


def pred_wnop(s_wnop):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_wnop = []
    for s_wnop1 in s_wnop:
        pr_wnop.append(s_wnop1.argmax().item())
    return pr_wnop


def pred_wc(wn, s_wc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    ! Returned index is sorted!
    """
    # get g_num
    pr_wc = []
    for b, wn1 in enumerate(wn):
        s_wc1 = s_wc[b]

        pr_wc1 = np.argsort(-s_wc1.data.cpu().numpy())[:wn1]
        pr_wc1.sort()

        pr_wc.append(list(pr_wc1))
    return pr_wc


def pred_wo(wn, s_wo):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # s_wo = [B, 4, n_op]
    pr_wo_a = s_wo.argmax(dim=2)  # [B, 4]
    # get g_num
    pr_wo = []
    for b, pr_wo_a1 in enumerate(pr_wo_a):
        wn1 = wn[b]
        pr_wo.append(list(pr_wo_a1.data.cpu().numpy()[:wn1]))

    return pr_wo


def pred_tags(s_tags):
    return (s_tags > 0.5).long().cpu().numpy()


def extract_val(pr_tags, l_n):
    value_indexes = []
    value_nums = []
    for b, tags in enumerate(pr_tags):
        vi = []
        num = 0
        start = -1
        for i in range(l_n[b]):
            t = tags[i]
            if t == 1:
                if start < 0:
                    start = i
            else:
                if start >= 0:
                    vi.append((start, i - 1))
                    num += 1
                    start = -1
        if start >= 0:
            vi.append((start, l_n[b] - 1))
            num += 1
        if not vi and num == 0:
            vi = [[0, 0]]
            num += 1
        value_indexes.append(vi)
        value_nums.append(num)
    return value_indexes, value_nums


def pred_wvi(wn, s_match, value_indexes, value_nums):
    """
    :s_match [bS, col_num, val_num]
    """
    wvis = []
    for b, s_match1 in enumerate(s_match):
        wvi = []
        # val_num = value_nums[b]
        for col_i in range(wn[b]):
            wvi1 = []
            s_match11 = s_match1[col_i]
            max_prob = 0.0
            max_vid = -1
            for vid, m in enumerate(s_match11):
                if m > 0.5:
                    st, ed = value_indexes[b][vid]
                    wvi1.append([st, ed])
                if m > max_prob:
                    max_prob = m
                    max_vid = vid
            if not wvi1:
                st, ed = value_indexes[b][max_vid]
                wvi1.append([st, ed])
            wvi.append(wvi1)
        wvis.append(wvi)
    return wvis



def convert_pr_wvi_to_string(pr_wvi, nlu_t):
    """
    - Convert to the string in whilte-space-separated tokens
    - Add-hoc addition.
    """
    pr_wv_str = []
    for b, pr_wvi1 in enumerate(pr_wvi):
        pr_wv_str1 = []
        nlu_t1 = nlu_t[b]
        for col, pr_wvi11 in enumerate(pr_wvi1):
            pr_wv_str11 = []
            for i_wn, pr_wvi111 in enumerate(pr_wvi11):
                st_idx, ed_idx = pr_wvi111

                # Ad-hoc modification of ed_idx to deal with wp-tokenization effect.
                # e.g.) to convert "butler cc (" ->"butler cc (ks)" (dev set 1st question).
                pr_wv_str111 = nlu_t1[st_idx:ed_idx+1]
                pr_wv_str11.append(pr_wv_str111)
            pr_wv_str1.append(pr_wv_str11)
        pr_wv_str.append(pr_wv_str1)
    return pr_wv_str

wnop_map_reverse = {
    0: (0, 1), 1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 1), 5: (2, 2), 6: (2, 3)
}


def get_wnop(wnop):
    l_wn, l_conn_op = [], []
    for nop in wnop:
        conn_op, wn = wnop_map_reverse[nop]
        l_wn.append(wn)
        l_conn_op.append(conn_op)
    return l_wn, l_conn_op


def pred_sw_se(s_sn, s_sc, s_sa, s_wnop, s_wc, s_wo, s_tags, s_match,
               l_n, g_sn=None, g_wnop=None, g_tags=None):
    pr_sn = pred_sn(s_sn)
    sn = g_sn if g_sn else pr_sn
    pr_sc = pred_sc(sn, s_sc)
    pr_sa = pred_sa(sn, s_sa)
    pr_wnop = pred_wnop(s_wnop)
    pr_wn, pr_conn_op = get_wnop(pr_wnop)
    wn = get_wnop(g_wnop)[0] if g_wnop else pr_wn
    pr_wc = pred_wc(wn, s_wc)
    pr_wo = pred_wo(wn, s_wo)
    pr_tags = pred_tags(s_tags)
    tags = g_tags if g_tags else pr_tags
    value_indexes, value_nums = extract_val(tags, l_n)
    pr_wvi = pred_wvi(wn, s_match, value_indexes, value_nums)

    return pr_sn, pr_sc, pr_sa, pr_wn, pr_conn_op, pr_wc, pr_wo, pr_tags, pr_wvi


def merge_wv_t1_eng(where_str_tokens, NLq):
    """
    Almost copied of SQLNet.
    The main purpose is pad blank line while combining tokens.
    """
    nlq = NLq.lower()
    where_str_tokens = [tok.lower() for tok in where_str_tokens]
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$'
    special = {'-LRB-': '(',
               '-RRB-': ')',
               '-LSB-': '[',
               '-RSB-': ']',
               '``': '"',
               '\'\'': '"',
               }
               # '--': '\u2013'} # this generate error for test 5661 case.
    ret = ''
    double_quote_appear = 0
    for raw_w_token in where_str_tokens:
        # if '' (empty string) of None, continue
        if not raw_w_token:
            continue

        # Change the special characters
        w_token = special.get(raw_w_token, raw_w_token)  # maybe necessary for some case?

        # check the double quote
        if w_token == '"':
            double_quote_appear = 1 - double_quote_appear

        # Check whether ret is empty. ret is selected where condition.
        if len(ret) == 0:
            pass
        # Check blank character.
        elif len(ret) > 0 and ret + ' ' + w_token in nlq:
            # Pad ' ' if ret + ' ' is part of nlq.
            ret = ret + ' '

        elif len(ret) > 0 and ret + w_token in nlq:
            pass  # already in good form. Later, ret + w_token will performed.

        # Below for unnatural question I guess. Is it likely to appear?
        elif w_token == '"':
            if double_quote_appear:
                ret = ret + ' '  # pad blank line between next token when " because in this case, it is of closing apperas
                # for the case of opening, no blank line.

        elif w_token[0] not in alphabet:
            pass  # non alphabet one does not pad blank line.

        # when previous character is the special case.
        elif (ret[-1] not in ['(', '/', '\u2013', '#', '$', '&']) and (ret[-1] != '"' or not double_quote_appear):
            ret = ret + ' '
        ret = ret + w_token

    return ret.strip()


def edit(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 2
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]


def post_process(cell_values, value):
    min_d = None
    min_c = -1
    for c in cell_values:
        if value in c.lower():
            ed = len(c) - 100000
            if min_d is None or min_d > ed:
                min_d = ed
                min_c = c
        else:
            ed = edit(c.lower(), value)
            if min_d is None or min_d > ed:
                min_d = ed
                min_c = c
    return min_d, min_c


def normalize_sql(sql, table):
    p_conds = sql['conds']
    for k, x in enumerate(p_conds):
        is_num = table['types'][x[0]] == 'real'
        try:
            float(x[2])
        except:
            is_num = False
        cell_values = []
        for cells in table['rows']:
            v = str(cells[x[0]])
            if v.endswith('.0'):
                v = v[:-2]
            cell_values.append(v)
        val = x[2]
        if not is_num:
            if val == '负':
                val = '0'
            else:
                ed, val = post_process(cell_values, val)
            if val.endswith('.0'):
                val = val[:-2]
        x[2] = val
        p_conds[k] = (x[0], x[1], x[2])
    if len(p_conds) == 1:
        sql['cond_conn_op'] = 0
    if len(p_conds) > 1 and sql['cond_conn_op'] == 0:
        sql['cond_conn_op'] = 2
    p_conds = sorted(list(set(p_conds)))
    sql['conds'] = p_conds


def get_train_acc(g_sql, p_sql, table, normalized=False):
    g_conds = g_sql['conds']
    for k, x in enumerate(g_conds):
        if x[2].endswith('.0'):
            x[2] = x[2][:-2]
        g_conds[k] = (x[0], x[1], x[2])
    g_conds.sort()
    if not normalized:
        p_sql = normalize_sql(p_sql, table)
    p_conds = p_sql['conds']

    sa = set(zip(p_sql['sel'], p_sql['agg'])) == set(zip(g_sql['sel'], g_sql['agg']))
    co = p_sql['cond_conn_op'] == g_sql['cond_conn_op']
    cond = g_conds == p_conds
    sql = sa and co and cond
    return sql


def get_acc(g_sql, p_sql, pr_wc, pr_wo, table, normalized=False):
    g_conds = g_sql['conds']
    for k, x in enumerate(g_conds):
        if x[2].endswith('.0'):
            x[2] = x[2][:-2]
        g_conds[k] = (x[0], x[1], x[2])
    g_conds.sort()
    if not normalized:
        p_sql = normalize_sql(p_sql, table)
    p_conds = p_sql['conds']

    sn = len(set(p_sql['sel'])) == len(set(g_sql['sel']))
    sc = set(p_sql['sel']) == set(g_sql['sel'])
    sa = set(zip(p_sql['sel'], p_sql['agg'])) == set(zip(g_sql['sel'], g_sql['agg']))
    co = p_sql['cond_conn_op'] == g_sql['cond_conn_op']
    wn = len(set(pr_wc)) == len(set(map(lambda x: x[0], g_conds)))
    wc = set(pr_wc) == set(map(lambda x: x[0], g_conds))
    wo = set(zip(pr_wc, pr_wo)) == set(map(lambda x: (x[0], x[1]), g_conds))
    wv = set(map(lambda x: (x[0], x[2]), p_conds)) == set(map(lambda x: (x[0], x[2]), g_conds))
    cond = g_conds == p_conds
    sql = sa and co and cond
    return sn, sc, sa, co, wn, wc, wo, wv, cond, sql


def get_acc_x(g_sql_i, p_sql_i, table, db_cursor):
    g_sel = g_sql_i['sel']
    g_agg = g_sql_i['agg']
    g_sa = list(zip(g_sel, g_agg))
    g_sa.sort()
    g_sql_i['sel'] = []
    g_sql_i['agg'] = []
    for s, a in g_sa:
        g_sql_i['sel'].append(s)
        g_sql_i['agg'].append(a)
    g_sql = generate_sql_q1(g_sql_i, table)
    p_sql = generate_sql_q1(p_sql_i, table)
    g = set(db_cursor.execute(g_sql))
    p = set(db_cursor.execute(p_sql))
    return g == p, list(p)


def generate_sql_i(pr_sc, pr_sa, pr_conn_op, pr_wc, pr_wo, pr_wv_str, nlu):
    pr_sql_i = []
    for b, nlu1 in enumerate(nlu):
        conds = []
        for i_wn in range(len(pr_wc[b])):
            for val in pr_wv_str[b][i_wn]:
                conds1 = []
                conds1.append(pr_wc[b][i_wn])
                conds1.append(pr_wo[b][i_wn])
                merged_wv11 = merge_wv_t1_eng(val, nlu[b])
                conds1.append(merged_wv11)
                conds.append(conds1)

        pr_sql_i1 = {'agg': pr_sa[b], 'sel': pr_sc[b],
                     'conds': conds, 'cond_conn_op': pr_conn_op[b]}
        pr_sql_i.append(pr_sql_i1)
    return pr_sql_i


def save_for_evaluation(path_save, results, dset_name, epoch, use_filename=False):
    if use_filename:
        path_save_file = path_save
    else:
        path_save_file = os.path.join(path_save, f'results_{dset_name}-{epoch}.jsonl')
    with open(path_save_file, 'w', encoding='utf-8') as f:
        for i, r1 in enumerate(results):
            json_str = json.dumps(r1, ensure_ascii=False, default=json_default_type_checker)
            json_str += '\n'

            f.writelines(json_str)
    return path_save_file


def sort_and_generate_pr_w(pr_sql_i):
    pr_wc = []
    pr_wo = []
    pr_wv = []
    for b, pr_sql_i1 in enumerate(pr_sql_i):
        conds1 = pr_sql_i1["conds"]
        pr_wc1 = []
        pr_wo1 = []
        pr_wv1 = []

        # Generate
        for i_wn, conds11 in enumerate(conds1):
            pr_wc1.append( conds11[0])
            pr_wo1.append( conds11[1])
            pr_wv1.append( conds11[2])

        # sort based on pr_wc1
        idx = np.argsort(pr_wc1)
        pr_wc1 = np.array(pr_wc1)[idx].tolist()
        pr_wo1 = np.array(pr_wo1)[idx].tolist()
        pr_wv1 = np.array(pr_wv1)[idx].tolist()

        conds1_sorted = []
        for i, idx1 in enumerate(idx):
            conds1_sorted.append( conds1[idx1] )


        pr_wc.append(pr_wc1)
        pr_wo.append(pr_wo1)
        pr_wv.append(pr_wv1)

        pr_sql_i1['conds'] = conds1_sorted

    return pr_wc, pr_wo, pr_wv, pr_sql_i


def generate_sql_q1(sql_i1, tb1):
    """
        sql = {'sel': 5, 'agg': 4, 'conds': [[3, 0, '59']]}
        agg_ops = ['', 'max', 'min', 'count', 'sum', 'avg']
        cond_ops = ['=', '>', '<', 'OP']

        Temporal as it can show only one-time conditioned case.
        sql_query: real sql_query
        sql_plus_query: More redable sql_query

        "PLUS" indicates, it deals with the some of db specific facts like PCODE <-> NAME
    """
    agg_ops = ['', 'AVG', 'MAX', 'MIN', 'COUNT', 'SUM']
    cond_ops = ['>', '<', '=', '!=']
    conn_ops = ['', 'AND', 'OR']

    headers = tb1["header"]
    types = tb1["types"]
    # select_header = headers[sql['sel']].lower()
    # try:
    #     select_table = tb1["name"]
    # except:
    #     print(f"No table name while headers are {headers}")
    select_table = tb1["id"]

    select_agg = [agg_ops[agg] for agg in sql_i1['agg']]
    select_header = [f'col_{col+1}' for col in sql_i1['sel']]
    # sql_query_part1 = f'SELECT {select_agg}({select_header}) '
    sql_query_part1 = 'SELECT ' + ', '.join([f'{agg}(`{col}`) ' if agg else f'`{col}` '
                                             for agg, col in zip(select_agg, select_header)])

    conn_op = conn_ops[sql_i1['cond_conn_op']]
    where_num = len(sql_i1['conds'])
    if where_num == 0:
        sql_query_part2 = f'FROM `Table_{select_table}`'
        # sql_query_part2 = f''
        # sql_plus_query_part2 = f'FROM {select_table}'

    else:
        sql_query_part2 = f'FROM `Table_{select_table}` WHERE'
        # sql_query_part2 = f'WHERE'
        # sql_plus_query_part2 = f'FROM {select_table_refined} WHERE'
        # ----------------------------------------------------------------------------------------------------------
        for i in range(where_num):
            # check 'OR'
            # number_of_sub_conds = len(sql['conds'][i])
            where_header_idx, where_op_idx, where_str = sql_i1['conds'][i]
            where_header = f'col_{where_header_idx+1}'
            col_type = types[where_header_idx]
            where_op = cond_ops[where_op_idx]
            if i > 0:
                if not conn_op:
                    conn_op = 'OR'
                sql_query_part2 += ' ' + conn_op
                # sql_plus_query_part2 += ' AND'
            is_number = col_type == 'real'
            try:
                float(where_str)
            except:
                is_number = False
            if not is_number:
                where_str = f'"{where_str}"'
            sql_query_part2 += f" `{where_header}` {where_op} {where_str}"

    sql_query = sql_query_part1 + sql_query_part2
    # sql_plus_query = sql_plus_query_part1 + sql_plus_query_part2

    return sql_query

