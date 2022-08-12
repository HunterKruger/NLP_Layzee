import math
import json
import re
import random
import os 
import numpy as np
import cn2an
from tqdm import tqdm_notebook as tqdm
from collections import defaultdict

import keras
from keras_bert import get_checkpoint_paths, load_vocabulary, Tokenizer, load_trained_model_from_checkpoint
from keras.utils.data_utils import Sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Lambda, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from nl2sql.utils import read_data, read_tables, SQL, Query, Question, Table


###  CONFIGURATION  ###

# GPU id setting, -1 to disable GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   
NUM_GPUS = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

# Test set
TEST_TABLE_FILE = '../../TableQA-master/test/test.tables.json'
TEST_DATA_FILE = '../../TableQA-master/test/test.json'

# Bert path
BERT_MODEL_PATH = '../../../experiments/model/chinese_wwm_L-12_H-768_A-12'
BERT_PATH = get_checkpoint_paths(BERT_MODEL_PATH)

BATCH_SIZE = NUM_GPUS * 32
INPUT_MODEL_PATH = '../model/m2.h5'  
TASK1_PREDICTION = '../result/task1_output.json'
FINAL_OUTPUT_FILE = '../result/task2_output.json'

###  CONFIGURATION  ###

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def cn_to_an(string):
    try:
        return str(cn2an.cn2an(string, 'normal'))
    except ValueError:
        return string

def an_to_cn(string):
    try:
        return str(cn2an.an2cn(string))
    except ValueError:
        return string

def str_to_num(string):
    try:
        float_val = float(cn_to_an(string))
        if int(float_val) == float_val:   
            return str(int(float_val))
        else:
            return str(float_val)
    except ValueError:
        return None

def str_to_year(string):
    year = string.replace('年', '')
    year = cn_to_an(year)
    if is_float(year) and float(year) < 1900:
        year = int(year) + 2000
        return str(year)
    else:
        return None
    
def load_json(json_file):
    result = []
    if json_file:
        with open(json_file) as file:
            for line in file:
                result.append(json.loads(line))
    return result

class QuestionCondPair:
    def __init__(self, query_id, question, cond_text, cond_sql, label):
        self.query_id = query_id
        self.question = question
        self.cond_text = cond_text
        self.cond_sql = cond_sql
        self.label = label

    def __repr__(self):
        repr_str = ''
        repr_str += 'query_id: {}\n'.format(self.query_id)
        repr_str += 'question: {}\n'.format(self.question)
        repr_str += 'cond_text: {}\n'.format(self.cond_text)
        repr_str += 'cond_sql: {}\n'.format(self.cond_sql)
        repr_str += 'label: {}\n'.format(self.label)
        return repr_str
    
class NegativeSampler:
    """
    从 question - cond pairs 中采样
    """
    def __init__(self, neg_sample_ratio=10):
        self.neg_sample_ratio = neg_sample_ratio
    
    def sample(self, data):
        positive_data = [d for d in data if d.label == 1]
        negative_data = [d for d in data if d.label == 0]
        negative_sample = random.sample(negative_data, 
                                        len(positive_data) * self.neg_sample_ratio)
        return positive_data + negative_sample
   
class FullSampler:
    """
    不抽样，返回所有的 pairs
    
    """
    def sample(self, data):
        return data

class CandidateCondsExtractor:
    """
    params:
        - share_candidates: 在同 table 同 column 中共享 real 型 candidates
    """
    CN_NUM = '〇一二三四五六七八九零壹贰叁肆伍陆柒捌玖貮两'
    CN_UNIT = '十拾百佰千仟万萬亿億兆点'
    
    def __init__(self, share_candidates=True):
        self.share_candidates = share_candidates
        self._cached = False
    
    def build_candidate_cache(self, queries):
        self.cache = defaultdict(set)
        print('building candidate cache')
        for query_id, query in tqdm(enumerate(queries), total=len(queries)):
            value_in_question = self.extract_values_from_text(query.question.text)
            
            for col_id, (col_name, col_type) in enumerate(query.table.header):
                value_in_column = self.extract_values_from_column(query, col_id)
                if col_type == 'text':
                    cond_values = value_in_column
                elif col_type == 'real':
                    if len(value_in_column) == 1: 
                        cond_values = value_in_column + value_in_question
                    else:
                        cond_values = value_in_question
                cache_key = self.get_cache_key(query_id, query, col_id)
                self.cache[cache_key].update(cond_values)
        self._cached = True
    
    def get_cache_key(self, query_id, query, col_id):
        if self.share_candidates:
            return (query.table.id, col_id)
        else:
            return (query_id, query.table.id, col_id)
        
    def extract_year_from_text(self, text):
        values = []
        num_year_texts = re.findall(r'[0-9][0-9]年', text)
        values += ['20{}'.format(text[:-1]) for text in num_year_texts]
        cn_year_texts = re.findall(r'[{}][{}]年'.format(self.CN_NUM, self.CN_NUM), text)
        cn_year_values = [str_to_year(text) for text in cn_year_texts]
        values += [value for value in cn_year_values if value is not None]
        return values
    
    def extract_num_from_text(self, text):
        values = []
        num_values = re.findall(r'[-+]?[0-9]*\.?[0-9]+', text)
        values += num_values
        
        cn_num_unit = self.CN_NUM + self.CN_UNIT
        cn_num_texts = re.findall(r'[{}]*\.?[{}]+'.format(cn_num_unit, cn_num_unit), text)
        cn_num_values = [str_to_num(text) for text in cn_num_texts]
        values += [value for value in cn_num_values if value is not None]
    
        cn_num_mix = re.findall(r'[0-9]*\.?[{}]+'.format(self.CN_UNIT), text)
        for word in cn_num_mix:
            num = re.findall(r'[-+]?[0-9]*\.?[0-9]+', word)
            for n in num:
                word = word.replace(n, an_to_cn(n))
            str_num = str_to_num(word)
            if str_num is not None:
                values.append(str_num)
        return values
    
    def extract_values_from_text(self, text):
        values = []
        values += self.extract_year_from_text(text)
        values += self.extract_num_from_text(text)
        return list(set(values))
   
    def extract_values_from_column(self, query, col_ids):
        question = query.question.text
        question_chars = set(query.question.text)
        unique_col_values = set(query.table.df.iloc[:, col_ids].astype(str))
        select_col_values = [v for v in unique_col_values 
                             if (question_chars & set(v))]
        return select_col_values
      
class QuestionCondPairsDataset:
    """
    question - cond pairs 数据集
    """
    OP_PATTERN = {
        'real':
        [
            {'cond_op_idx': 0, 'pattern': '{col_name}大于{value}'},
            {'cond_op_idx': 1, 'pattern': '{col_name}小于{value}'},
            {'cond_op_idx': 2, 'pattern': '{col_name}是{value}'}
        ],
        'text':
        [
            {'cond_op_idx': 2, 'pattern': '{col_name}是{value}'}
        ]
    }    
    
    def __init__(self, queries, candidate_extractor, has_label=True, model_1_outputs=None):
        self.candidate_extractor = candidate_extractor
        self.has_label = has_label
        self.model_1_outputs = model_1_outputs
        self.data = self.build_dataset(queries)
        
    def build_dataset(self, queries):
        if not self.candidate_extractor._cached:
            self.candidate_extractor.build_candidate_cache(queries)
            
        pair_data = []
        for query_id, query in enumerate(queries):
            select_col_id = self.get_select_col_id(query_id, query)
            for col_id, (col_name, col_type) in enumerate(query.table.header):
                if col_id not in select_col_id:
                    continue
                    
                cache_key = self.candidate_extractor.get_cache_key(query_id, query, col_id)
                values = self.candidate_extractor.cache.get(cache_key, [])
                pattern = self.OP_PATTERN.get(col_type, [])
                pairs = self.generate_pairs(query_id, query, col_id, col_name, 
                                               values, pattern)
                pair_data += pairs
        return pair_data
    
    def get_select_col_id(self, query_id, query):
        if self.model_1_outputs:
            select_col_id = [cond_col for cond_col, *_ in self.model_1_outputs[query_id]['conds']]
        elif self.has_label:
            select_col_id = [cond_col for cond_col, *_ in query.sql.conds]
        else:
            select_col_id = list(range(len(query.table.header)))
        return select_col_id
            
    def generate_pairs(self, query_id, query, col_id, col_name, values, op_patterns):
        pairs = []
        for value in values:
            for op_pattern in op_patterns:
                cond = op_pattern['pattern'].format(col_name=col_name, value=value)
                cond_sql = (col_id, op_pattern['cond_op_idx'], value)
                real_sql = {}
                if self.has_label:
                    real_sql = {tuple(c) for c in query.sql.conds}
                label = 1 if cond_sql in real_sql else 0
                pair = QuestionCondPair(query_id, query.question.text,
                                        cond, cond_sql, label)
                pairs.append(pair)
        return pairs
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class SimpleTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R
          
def construct_model(paths=BERT_PATH):
    token_dict = load_vocabulary(paths.vocab)
    tokenizer = SimpleTokenizer(token_dict)

    bert_model = load_trained_model_from_checkpoint(
        paths.config, paths.checkpoint, seq_len=None)
    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,), name='input_x1', dtype='int32')
    x2_in = Input(shape=(None,), name='input_x2')
    x = bert_model([x1_in, x2_in])
    x_cls = Lambda(lambda x: x[:, 0])(x)
    y_pred = Dense(1, activation='sigmoid', name='output_similarity')(x_cls)

    model = Model([x1_in, x2_in], y_pred)

    return model, tokenizer

class QuestionCondPairsDataseq(Sequence):
    def __init__(self, dataset, tokenizer, is_train=True, max_len=120, 
                 sampler=None, shuffle=False, batch_size=32):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_len = max_len
        self.sampler = sampler
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()       
    
    def _pad_sequences(self, seqs, max_len=None):
        return pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')
    
    def __getitem__(self, batch_id):
        batch_data_indices = \
            self.global_indices[batch_id * self.batch_size: (batch_id + 1) * self.batch_size]
        batch_data = [self.data[i] for i in batch_data_indices]

        X1, X2 = [], []
        Y = []
        
        for data in batch_data:
            x1, x2 = self.tokenizer.encode(first=data.question.lower(), 
                                           second=data.cond_text.lower())
            X1.append(x1)
            X2.append(x2)
            if self.is_train:
                Y.append([data.label])
    
        X1 = self._pad_sequences(X1, max_len=self.max_len)
        X2 = self._pad_sequences(X2, max_len=self.max_len)
        inputs = {'input_x1': X1, 'input_x2': X2}
        if self.is_train:
            Y = self._pad_sequences(Y, max_len=1)
            outputs = {'output_similarity': Y}
            return inputs, outputs
        else:
            return inputs
                    
    def on_epoch_end(self):
        self.data = self.sampler.sample(self.dataset)
        self.global_indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.global_indices)
    
    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

def merge_result(qc_pairs, result, threshold):
    select_result = defaultdict(set)
    for pair, score in zip(qc_pairs, result):
        if score > threshold:
            select_result[pair.query_id].update([pair.cond_sql])
    return dict(select_result)

def main():

    # read data
    test_tables = read_tables(TEST_TABLE_FILE)
    test_data = read_data(TEST_DATA_FILE, test_tables)

    task1_result = load_json(TASK1_PREDICTION)

    model, tokenizer = construct_model(BERT_PATH)

    model.load_weights(INPUT_MODEL_PATH)

    test_qc_pairs = QuestionCondPairsDataset(
        test_data, 
        candidate_extractor=CandidateCondsExtractor(share_candidates=True),
        has_label=False,
        model_1_outputs=task1_result
    )

    test_qc_pairs_seq = QuestionCondPairsDataseq(
        test_qc_pairs, 
        tokenizer,                 
        sampler=FullSampler(), shuffle=False, batch_size=BATCH_SIZE
    )


    if NUM_GPUS > 1:
        print('using {} gpus'.format(NUM_GPUS))
        model = multi_gpu_model(model, gpus=NUM_GPUS)
    else:
        print('Use single gpu or cpu')


    test_result = model.predict_generator(test_qc_pairs_seq, verbose=1)

    task2_result = merge_result(test_qc_pairs, test_result, threshold=0.995)   

    with open(FINAL_OUTPUT_FILE, 'w') as f:
        for query_id, pred_sql in enumerate(task1_result):
            cond = list(task2_result.get(query_id, []))
            pred_sql['conds'] = cond
            json_str = json.dumps(pred_sql, ensure_ascii=False)
            f.write(json_str + '\n')


if __name__ == "__main__":
    # execute only if run as a script
    main()