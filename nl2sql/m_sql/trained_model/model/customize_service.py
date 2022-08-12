# -*- coding: utf-8 -*-
import json
import random
import sqlite3

import rouge
import torch
from collections import OrderedDict
import torch.backends.cudnn as cudnn

import sys, os

from transformers import BertTokenizer, BertModel, BertConfig
ParentClass = None
try:
    from model_service.pytorch_model_service import PTServingBaseService as ParentClass
except:
    ParentClass = object

from model import Seq2SQL_v1
from sql_utils.utils_tableqa import get_wemb_bert, pred_sw_se, convert_pr_wvi_to_string, \
    generate_sql_i, normalize_sql, generate_sql_q1

sys.path.insert(0, os.path.dirname(__file__))


cudnn.benchmark = True


class ModelClass(ParentClass):
    def __init__(self, model_name, model_path):
        """
        :param model_name: 本参数必须保留，随意传入一个字符串值即可
        :param model_path: 模型所在的路径，比如 xxx/xxx.h5、xxx/xxx.pth，如果在ModelArts中运行，该参数会自动传入，不需要人工指定
        """
        self.model_name = model_name  # 本行代码必须保留，且无需修改
        self.model_path = model_path  # 本行代码必须保留，且无需修改

        self.cpu = not torch.cuda.is_available()
        self.input_size = 768
        self.dropout_rate = 0
        self.device = torch.device("cpu" if self.cpu else "cuda")

        self.input_path = "./model/"
        # self.input_path = "./"
        self.db_path = os.path.join(self.input_path, 'val.db')
        self.table_path = os.path.join(self.input_path, 'val.tables.jsonl')
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.table_dict = self.get_table_dict(self.table_path)

        print('Loading model from %s' % model_path)
        torch.set_grad_enabled(False)

        self.model = Seq2SQL_v1(self.input_size, self.dropout_rate)
        self.bert_path = os.path.join(self.input_path, 'bert_config')
        self.model_bert = BertModel(BertConfig.from_pretrained(self.bert_path))
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_config = self.model_bert.config

        if self.cpu:
            pretrained_dict = torch.load(model_path, map_location='cpu')
        else:
            pretrained_dict = torch.load(model_path)

        self.model_bert.load_state_dict(pretrained_dict['model_bert'])
        self.model_bert.to(self.device)
        self.model_bert.eval()
        self.model.load_state_dict(pretrained_dict['model'])
        self.model.to(self.device)
        self.model.eval()
        print('load model success')

    def get_table_dict(self, table_path):
        table_dict = {}
        with open(table_path, 'r', encoding='utf-8') as f:
            for line in f:
                table = json.loads(line)
                table_dict[table['id']] = table
                cell_values = [list(map(lambda x: x[i], table['rows']))
                               for i, _ in enumerate(table['header'])]
                for i, vals in enumerate(cell_values):
                    if table['types'][i] == 'real':
                        cell_values[i] = list(set(map(lambda v: str(v)[:-2]
                        if str(v).endswith('.0') else str(v), vals)))
                    else:
                        cell_values[i] = list(set(map(str.lower, vals)))
                table['cell_values'] = cell_values
        return table_dict

    @staticmethod
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

    def _preprocess(self, data):
        return data

    def preprocess(self, data, tables):
        nlu = data['question'].lower()
        nlu_t = self.process_bert_tokens(self.tokenizer.tokenize(nlu))
        tid = data['table_id']
        table = tables[tid]

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

        assert len(col_v) == len(table['header'])
        hs1 = []
        for h, v in zip(table['header'], col_v):
            if len(v) > 50:
                v = 'none'
            hs1.append(h + ',' + v)

        return nlu, nlu_t, tid, table, hs1

    def generate_sql(self, sql_i1, tb1):
        agg_ops = ['', 'AVG', 'MAX', 'MIN', 'COUNT', 'SUM']
        cond_ops = ['>', '<', '=', '!=']
        conn_ops = ['', 'AND', 'OR']

        headers = tb1["header"]
        types = tb1["types"]

        select_agg = [agg_ops[agg] for agg in sql_i1['agg']]
        select_header = [f'{headers[col]}' for col in sql_i1['sel']]
        sql_query_part1 = 'SELECT ' + ', '.join([f'{agg}(`{col}`) ' if agg else f'`{col}` '
                                                 for agg, col in zip(select_agg, select_header)])

        conn_op = conn_ops[sql_i1['cond_conn_op']]
        where_num = len(sql_i1['conds'])
        if where_num == 0:
            sql_query_part2 = f''
        else:
            sql_query_part2 = f'WHERE'
            for i in range(where_num):

                where_header_idx, where_op_idx, where_str = sql_i1['conds'][i]
                where_header = f'{headers[where_header_idx]}'
                col_type = types[where_header_idx]
                where_op = cond_ops[where_op_idx]
                if i > 0:
                    if not conn_op:
                        conn_op = 'OR'
                    sql_query_part2 += ' ' + conn_op
                is_number = col_type == 'real'
                try:
                    float(where_str)
                except:
                    is_number = False
                if not is_number:
                    where_str = f'"{where_str}"'
                sql_query_part2 += f" `{where_header}` {where_op} {where_str}"

        sql_query = sql_query_part1 + sql_query_part2
        return sql_query


    def _inference(self, data):

        nlu, nlu_t, tid, table, hs = self.preprocess(data, self.table_dict)

        wemb_cls, wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(self.bert_config, self.model_bert, self.tokenizer,
                            [nlu_t], [hs], 512)

        l_n_t = []
        for t in t_to_tt_idx:
            l_n_t.append(len(t))

        s_sn, s_sc, s_sa, s_wnop, s_wc, s_wo, s_tags, s_match = \
            self.model(wemb_cls, wemb_n, l_n_t, wemb_h, l_hpu, l_hs, t_to_tt_idx)
        # prediction
        pr_sn, pr_sc, pr_sa, pr_wn, pr_conn_op, \
        pr_wc, pr_wo, pr_tags, pr_wvi = pred_sw_se(s_sn, s_sc, s_sa, s_wnop, s_wc, s_wo, s_tags, s_match, l_n_t)

        pr_wv_str = convert_pr_wvi_to_string(pr_wvi, [nlu_t])
        pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_conn_op, pr_wc, pr_wo, pr_wv_str, [nlu])[0]
        normalize_sql(pr_sql_i, table)
        try:
            sql = self.generate_sql(pr_sql_i, table)
        except:
            sql = 'None'
        try:
            sql_to_execute = generate_sql_q1(pr_sql_i, table)
            ex_result = list(self.cursor.execute(sql_to_execute))
        except:
            ex_result = []

        result = OrderedDict()
        result['sql'] = sql
        result['result'] = ex_result
        return result


if __name__ == '__main__':
    model_path = r'./model_epoch_17.pth'
    my_model = ModelClass('', model_path)
    data = {
        "question": "近四周成交量小于3574套并且环比低于69.7%的城市有几个",
        "table_id": "252c7b6b302e11e995ee542696d6e445"
    }
    data = my_model._preprocess(data)
    result = my_model._inference(data)
    print(json.dumps(dict(result), ensure_ascii=False, indent=2))
