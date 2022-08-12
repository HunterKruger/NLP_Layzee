import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import json
import rouge
from tqdm import tqdm
import copy
from transformers import BertTokenizer
import shutil

from sql_utils.utils_tableqa import extract_val
from sql_utils.utils_data_process import *
try:
    import moxing as mox
except:
    mox = None

def get_new_question_test(question, table):
    q = question.lower()
    tab_cells = table['values']
    edit_indexes = []
    vals = get_match_values(q, tab_cells)
    vals2 = get_values(q)
    edit_indexes += vals
    edit_indexes += vals2

    edit_indexes.sort(key=lambda x: (x[-1], -x[2]))
    new_edit_indexes = []
    for index in edit_indexes:
        if index[0] < 0.6:
            new_edit_indexes.append(index)
    edit_indexes = new_edit_indexes
    new_q = q
    _, final_val_indexes = get_new_q(new_q, edit_indexes)
    vals2_v = list(map(lambda x: x[3], vals2))
    val_indexes = []
    for k in final_val_indexes:
        if k[2] not in vals2_v:
            val_indexes.append(k)
    new_q = fix_chinese_num(new_q, val_indexes)
    new_q, change = process_question_number(new_q)
    new_q = fix_numbers(new_q, val_indexes)
    new_q = remove_dian(new_q)
    new_q = new_q.lower()
    return new_q


def get_value_map(question, table, sql):
    q = question.lower()
    tab_cells = table['values']

    edit_indexes = []
    vals = get_match_values(q, tab_cells)
    vals2 = get_values(q)
    edit_indexes += vals
    edit_indexes += vals2

    edit_indexes.sort(key=lambda x: (x[-1], -x[2]))
    new_edit_indexes = []
    for index in edit_indexes:
        if index[0] < 0.6:
            new_edit_indexes.append(index)
    edit_indexes = new_edit_indexes
    new_q = q
    _, final_val_indexes = get_new_q(new_q, edit_indexes)
    vals2_v = list(map(lambda x: x[3], vals2))
    val_indexes = []
    for k in final_val_indexes:
        if k[2] not in vals2_v:
            val_indexes.append(k)
    new_q = fix_chinese_num(new_q, val_indexes)
    new_q, change = process_question_number(new_q)
    new_q = fix_numbers(new_q, val_indexes)
    new_q = remove_dian(new_q)

    new_q = new_q.lower()
    values = []
    value_map = []
    is_not_in = False
    for k, (_, _, v) in enumerate(sql['conds']):
        is_not_in = False
        kv = v.strip().lower()
        is_number = False
        try:
            float(kv)
            is_number = True
            if kv.endswith('.0'):
                kv = kv[:-2]
            values.append(kv.strip().lower())
        except:
            values.append(kv.strip().lower())
        if kv in chinese_num_map:
            kv = str(chinese_num_map[v])
            is_number = True
        if v in new_q:
            kv = v
        if kv in new_q:
            st = new_q.index(kv)
            ed = st + len(kv)
            value_map.append((k, v, kv, st, ed))
        elif is_number:
            number_in = False
            try:
                p = int(kv)
                kvs = map(str, [p, p - 1, p + 1, p // 10000, p // 1000,
                                p * 10000, p * 1000, p * 100, p / 100])
            except:
                p = float(kv)
                kvs = map(str, [p, int(p), int(p) + 1, int(p * 100)])
            if '.' in kv:
                y1, m1 = kv.split('.')
                y1 = int(y1)
                m1 = int(m1)
                dates, index = get_year_month(new_q)
                if (y1, m1) in dates:
                    number_in = True
                    st, ed = index[dates.index((y1, m1))]
                    kv = new_q[st:ed]
                    value_map.append((k, v, kv, st, ed))
            else:
                for vv in kvs:
                    if vv in new_q:
                        st = new_q.index(vv)
                        ed = st + len(vv)
                        value_map.append((k, v, vv, st, ed))
                        number_in = True
                        break
            if not number_in:
                is_not_in = True
        else:
            len_q = len(new_q)
            len_v = len(kv)
            ee = edit(kv, new_q)
            index, len_p, min_e = find_min_str(0, len_q, ee, new_q, kv)
            if min_e / len_v >= 0.95:
                index, min_e, len_p = -1, len_q + 1, -1

            for j in range(0, len_q - len(kv) + 1):
                ee = edit(kv, new_q[j:j + len_v])
                if ee <= min_e and ee / len_v < 0.95:
                    index, len_p, min_e = find_min_str(j, len_v + 1, ee, new_q, kv)
            if index == -1:
                is_not_in = True
            else:
                value_map.append((k, v, new_q[index:index + len_p], index, index + len_p))
        if is_not_in:
            break
    if is_not_in:
        value_map = []
    return new_q, value_map


def can_add(st, ed, value_map):
    for i, (st1, ed1, v) in enumerate(value_map):
        if ed <= st1 or st >= ed1:
            continue
        else:
            return False
    return True


def get_new_question_train(question, table, sql):
    new_q, value_map = get_value_map(question, table, sql)
    need_process = False
    for i, (idx, nv, v, st, ed) in enumerate(value_map):
        if nv != v:
            need_process = True
            break
    if not need_process or not value_map:
        return new_q
    else:
        value_map = sorted(value_map, key=lambda x: x[3] - x[4])
        value_map_to_add = []
        for i, (idx, nv, v, st, ed) in enumerate(value_map):
            if can_add(st, ed, value_map_to_add):
                value_map_to_add.append((st, ed, nv))
        value_map_to_add.sort()
        q = new_q
        new_q = ''
        old_ed = 0
        for a, b, v in value_map_to_add:
            new_q += q[old_ed:a] + v
            old_ed = b
        if old_ed < len(q):
            new_q += q[old_ed:]
        return new_q


def count_lines(fname):
    with open(fname) as f:
        return sum(1 for _ in f)


def tokenize(sentence, tokenizer, lower=False):
    return tokenizer.tokenize(sentence.lower() if lower else sentence)


def process_bert_tokens(tokens):
    new_sequence = []
    cache = ''
    for tok in tokens:
        if tok.startswith('##') and len(tok) > 2:
            cache += tok[2:]
        else:
            if cache:
                new_sequence.append(cache)
            cache = tok
    if cache:
        new_sequence.append(cache)
    return new_sequence


def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    sl_s = ''.join(sl)
    try:
        float(sl_s)
        is_num = True
    except:
        is_num = False
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl or (not is_num and sl_s in ''.join(l[ind:ind + sll])):
            results.append((ind, ind + sll - 1))

    return results


def check_wv_tok_in_nlu_tok(wv_tok1, nlu_t1):
    """
    Jan.2019: Wonseok
    Generate SQuAD style start and end index of wv in nlu. Index is for of after WordPiece tokenization.

    Assumption: where_str always presents in the nlu.

    return:
    st_idx of where-value string token in nlu under CoreNLP tokenization scheme.
    """
    g_wvi1_corenlp = []
    nlu_t1_low = [tok.lower() for tok in nlu_t1]
    for i_wn, wv_tok11 in enumerate(wv_tok1):
        wv_tok11_low = [tok.lower() for tok in wv_tok11]
        results = find_sub_list(wv_tok11_low, nlu_t1_low)
        assert results
        st_idx, ed_idx = -1, -1
        if len(results) > 1:
            for a, b in results:
                for x, y in g_wvi1_corenlp:
                    if a > y or b < x:
                        st_idx, ed_idx = a, b
                        break
            if st_idx < 0:
                st_idx, ed_idx = results[0]
        else:
            st_idx, ed_idx = results[0]

        g_wvi1_corenlp.append( [st_idx, ed_idx] )

    return g_wvi1_corenlp


def get_value_match(ann, nlu, tokenizer, table):
    # db content utilization
    r = rouge.Rouge(['rouge-l'])
    cell_values = table['cell_values']
    col_v = []
    for i, cell_value in enumerate(cell_values):
        vals = []
        for cv in cell_value:
            try:
                score = r.get_scores(' '.join(nlu), ' '.join(cv))[0]['rouge-l']['f']
                vals.append((cv, score))
            except:
                pass
        random.shuffle(vals)
        vals.sort(key=lambda x: -x[1])
        col_v.append(vals[0][0])
    ann['col_value'] = col_v

    conds1 = ann['sql']['conds']
    wv_ann1 = []
    for conds11 in conds1:
        _wv_ann1 = tokenize(str(conds11[2]), tokenizer)
        _wv_ann1 = process_bert_tokens(_wv_ann1)
        wv_ann11 = _wv_ann1
        wv_ann1.append(wv_ann11)

    # Check whether wv_ann exsits inside question_tok

    wvi1_corenlp = check_wv_tok_in_nlu_tok(wv_ann1, ann['question_tok'])
    sorted_wvi1_corenlp = sorted(wvi1_corenlp)
    value_tags = [0 for _ in range(len(ann['question_tok']))]
    value_match = []
    for item in wvi1_corenlp:
        a, b = item
        for i in range(a, b + 1):
            value_tags[i] = 1
    value_index = extract_val([value_tags], [len(value_tags)])[0][0]

    for k, item in enumerate(wvi1_corenlp):
        a, b = item
        index = -1
        for i, (st, ed) in enumerate(value_index):
            if a >= st and b <= ed:
                index = i
                break
        assert index != -1
        wvi1_corenlp[k] = value_index[index]
        value_match.append(index)
    ann['where_val_index'] = wvi1_corenlp
    ann['value_tags'] = value_tags
    ann['value_match'] = value_match


def annotate_example_ws(example, table, bert_path, split, sql=None):
    """
    Jan. 2019: Wonseok
    Annotate only the information that will be used in our model.
    """

    tokenizer = BertTokenizer.from_pretrained(bert_path)
    ann = {'table_id': example['table_id']}
    if split == 'train':
        nlu = example['question']
    else:
        nlu = get_new_question_test(example['question'], table)
    _nlu_ann = tokenize(nlu, tokenizer)
    _nlu_ann = process_bert_tokens(_nlu_ann)
    ann['question'] = nlu
    ann['question_tok'] = _nlu_ann
    if not sql:
        ann['sql'] = example['sql']
    else:
        ann['sql'] = sql
    try:
        get_value_match(ann, nlu, tokenizer, table)
    except:
        if split == 'train':
            try:
                nlu = get_new_question_train(example['question'], table, ann['sql'])
                _nlu_ann = tokenize(nlu, tokenizer)
                _nlu_ann = process_bert_tokens(_nlu_ann)
                ann['question'] = nlu
                ann['question_tok'] = _nlu_ann
                get_value_match(ann, nlu, tokenizer, table)
            except:
                ann['where_val_index'] = [[0, 1]]
                ann['value_tags'] = [0 for _ in range(len(ann['question_tok']))]
                ann['value_match'] = [0]
                ann['tok_error'] = '-- value & question mismatch --'
        else:
            ann['where_val_index'] = [[0, 1]]
            ann['value_tags'] = [0 for _ in range(len(ann['question_tok']))]
            ann['value_match'] = [0]
            ann['tok_error'] = '-- value & question mismatch --'

    return ann


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_url", default=None, type=str)
    parser.add_argument("--train_url", default=None, type=str)

    parser.add_argument('--din', default='./data_and_model/tableqa/', help='data directory')
    parser.add_argument('--dout', default='./data_and_model/tableqa_tok_4/', help='output directory')
    parser.add_argument('--split', default='test', help='comma=separated list of splits to process')
    parser.add_argument('--answer_toy', default='False', type=str)
    parser.add_argument('--toy_size', default=10, type=int)
    parser.add_argument('--bert_path',
                        default='/mnt/d/nlp/pretrained_models/chinese-bert_chinese_wwm_L-12_H-768_A-12',
                        help='bert model path')
    args, _ = parser.parse_known_args()
    use_url = args.data_url and args.train_url
    obs = use_url and '://' in args.data_url and '://' in args.train_url

    answer_toy = args.answer_toy == 'True'
    toy_size = args.toy_size

    din = '/home/work/modelarts/inputs/'
    obs_out_path = None
    if mox and obs:
        mox.file.copy_parallel(args.data_url, din)
        args.din = din
        obs_out_path = args.train_url
        args.dout = '/home/work/modelarts/outputs/'
        bert_path = '/home/work/modelarts/inputs/bert/'
        mox.file.copy_parallel(args.bert_path, bert_path)
        args.bert_path = bert_path
    if not obs and use_url:
        args.din = args.data_url
        args.dout = args.train_url

    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    # for split in ['train', 'dev', 'test']:
    for split in args.split.split(','):
        if split == 'final' or split == 'final_test':
            fsplit = os.path.join(os.path.join(args.din, 'final'), 'final_test') + '.json'
            ftable = os.path.join(os.path.join(args.din, 'final'), 'final_test') + '.tables.json'
            fdb = os.path.join(os.path.join(args.din, 'final'), 'final_test') + '.db'
            fans = os.path.join(os.path.join(args.din, 'final'), 'standard.json')
            fout = os.path.join(args.dout, 'final_test') + '_tok.jsonl'
        else:
            fsplit = os.path.join(os.path.join(args.din, split), split) + '.json'
            ftable = os.path.join(os.path.join(args.din, split), split) + '.tables.json'
            fdb = os.path.join(os.path.join(args.din, split), split) + '.db'
            fans = os.path.join(os.path.join(args.din, split), 'standard.json')
            fout = os.path.join(args.dout, split) + '_tok.jsonl'

        if mox and obs:
            mox.file.copy_parallel(ftable, obs_out_path + split + '.tables.jsonl')
            mox.file.copy_parallel(fdb, obs_out_path + split + '.db')
        else:
            shutil.copy(ftable, os.path.join(args.dout, split) + '.tables.jsonl')
            shutil.copy(fdb, os.path.join(args.dout, split) + '.db')

        print('annotating {}'.format(fsplit))
        with open(fsplit) as fs, open(ftable) as ft, open(fout, 'wt') as fo:
            if 'test' in split:
                fa = open(fans)
            print('loading tables')

            # ws: Construct table dict with table_id as a key.
            tables = {}
            for line in tqdm(ft, total=count_lines(ftable)):
                table = json.loads(line)
                tables[table['id']] = table
                cell_values = [list(map(lambda x: x[i], table['rows']))
                               for i, _ in enumerate(table['header'])]
                for i, vals in enumerate(cell_values):
                    if table['types'][i] == 'real':
                        cell_values[i] = list(set(map(lambda v: str(v)[:-2]
                        if str(v).endswith('.0') else str(v), vals)))
                    else:
                        cell_values[i] = list(set(map(str.lower, vals)))
                table['cell_values'] = cell_values
                values = set()
                for cells in table['rows']:
                    for v in cells:
                        values.add(str(v)[:-2].lower() if str(v).endswith('.0')
                                   else str(v).lower())
                table['values'] = values

            print('loading examples')
            cnt = 0
            for line in tqdm(fs, total=count_lines(fsplit)):
                d = json.loads(line)
                if 'test' in split:
                    sql = json.loads(fa.readline())
                else:
                    sql = None
                # a = annotate_example(d, tables[d['table_id']])
                a = annotate_example_ws(d, tables[d['table_id']],
                                        bert_path=args.bert_path,
                                        split=split,
                                        sql=sql)
                if split == 'train' and 'tok_error' in a:
                    continue
                else:
                    fo.write(json.dumps(a, ensure_ascii=False) + '\n')
                    cnt += 1

                if answer_toy:
                    if cnt >= toy_size:
                        break
            print('wrote {} examples'.format(cnt))
        if mox and obs:
            mox.file.copy_parallel(fout, obs_out_path + split + '_tok.jsonl')
