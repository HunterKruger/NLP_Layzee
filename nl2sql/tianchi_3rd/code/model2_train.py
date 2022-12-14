### TRAINING MODEL2 ###

import math
import json
import re
import os 
import random
import cn2an
import numpy as np
from collections import defaultdict
from tqdm import tqdm_notebook as tqdm

import keras
from keras_bert import get_checkpoint_paths, load_vocabulary, Tokenizer, load_trained_model_from_checkpoint
from keras.utils.data_utils import Sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Lambda, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.callbacks import Callback, ModelCheckpoint,EarlyStopping
from nl2sql.utils import read_data, read_tables, SQL, Query, Question, Table

###  CONFIGURATION  ###

# GPU id setting, -1 to disable GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'   
NUM_GPUS = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

# CPU multi-threading
WORKERS = 4     

# Training set
TRAIN_TABLE_FILE = '../../TableQA-master/train/train.tables.json'
TRAIN_DATA_FILE = '../../TableQA-master/train/train.json'

# Bert path
BERT_MODEL_PATH = '../../../experiments/model/chinese_wwm_L-12_H-768_A-12'
BERT_PATH = get_checkpoint_paths(BERT_MODEL_PATH)

# Training params
LR = 1e-5
EPOCH = 1
BATCH_SIZE = NUM_GPUS * 32
CONTINUE = False                            # Training from ckpt

# model path
OUTPUT_MODEL_PATH = '../model/m2_spd.h5'    # output model weight path
INPUT_MODEL_PATH = '../model/m2.h5'         # input model weight path

###  CONFIGURATION  ###

# task1_file = '../result/task1_output.json'  # model1 prediction path


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
    year = string.replace('???', '')
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
    ??? question - cond pairs ?????????
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
    ??????????????????????????? pairs
    
    """
    def sample(self, data):
        return data

class CandidateCondsExtractor:
    """
    params:
        - share_candidates: ?????? table ??? column ????????? real ??? candidates
    """
    CN_NUM = '??????????????????????????????????????????????????????????????????'
    CN_UNIT = '????????????????????????????????????'
    
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
        num_year_texts = re.findall(r'[0-9][0-9]???', text)
        values += ['20{}'.format(text[:-1]) for text in num_year_texts]
        cn_year_texts = re.findall(r'[{}][{}]???'.format(self.CN_NUM, self.CN_NUM), text)
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
    question - cond pairs ?????????
    """
    OP_PATTERN = {
        'real':
        [
            {'cond_op_idx': 0, 'pattern': '{col_name}??????{value}'},
            {'cond_op_idx': 1, 'pattern': '{col_name}??????{value}'},
            {'cond_op_idx': 2, 'pattern': '{col_name}???{value}'}
        ],
        'text':
        [
            {'cond_op_idx': 2, 'pattern': '{col_name}???{value}'}
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



def main():

    # read data
    train_tables = read_tables(TRAIN_TABLE_FILE)
    train_data = read_data(TRAIN_DATA_FILE, train_tables)

    # build data set
    train_qc_pairs = QuestionCondPairsDataset(
        train_data, 
        candidate_extractor=CandidateCondsExtractor(share_candidates=False)
    )

    # build model
    model, tokenizer = construct_model(BERT_PATH)

    if CONTINUE:
        model.load_weights(INPUT_MODEL_PATH)

    if NUM_GPUS>1:
        print('using {} gpus'.format(NUM_GPUS))
        para_model = multi_gpu_model(model, gpus=NUM_GPUS)
        para_model.compile(
            loss={'output_similarity': 'binary_crossentropy'},
            optimizer=Adam(LR),
            metrics={'output_similarity': 'accuracy'}
        )
    else:
        print('using single gpu or cpu')
        model.compile(
            loss={'output_similarity': 'binary_crossentropy'},
            optimizer=Adam(LR),
            metrics={'output_similarity': 'accuracy'}
        )    

    # build data sequence
    train_qc_pairs_seq = QuestionCondPairsDataseq(
        train_qc_pairs, 
        tokenizer, 
        sampler=NegativeSampler(), 
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # callbacks = [EarlyStopping(monitor='loss',  patience=1, verbose=0, mode='max', restore_best_weights=True)]
    # ModelCheckpoint(filepath=model_path, monitor='loss', mode='max', save_best_only=True, save_weights_only=True)
    
    if NUM_GPUS>1:
        para_model.fit_generator(train_qc_pairs_seq, epochs=EPOCH, workers=WORKERS)
    else:
        model.fit_generator(train_qc_pairs_seq, epochs=EPOCH, workers=WORKERS)

    print('Saving weights...')
    model.save_weights(OUTPUT_MODEL_PATH)


if __name__ == "__main__":
    # execute only if run as a script
    main()