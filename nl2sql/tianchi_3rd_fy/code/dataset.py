"""
Created by FENG YUAN on 2021/10/6
"""

import re
import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
from utils import SQL

import config

class SqlLabelEncoder:
    """
    Convert SQL object into training labels.
    """
    def encode(self, sql: SQL, num_cols):
        cond_conn_op_label = sql.cond_conn_op
        
        sel_agg_label = np.ones(num_cols, dtype='int32') * len(SQL.agg_sql_dict)
        for col_id, agg_op in zip(sql.sel, sql.agg):
            if col_id < num_cols:
                sel_agg_label[col_id] = agg_op
            
        cond_op_label = np.ones(num_cols, dtype='int32') * len(SQL.op_sql_dict)
        for col_id, cond_op, _ in sql.conds:
            if col_id < num_cols:
                cond_op_label[col_id] = cond_op
            
        return cond_conn_op_label, sel_agg_label, cond_op_label
    
    def decode(self, cond_conn_op_label, sel_agg_label, cond_op_label):
        cond_conn_op = int(cond_conn_op_label)
        sel, agg, conds = [], [], []

        for col_id, (agg_op, cond_op) in enumerate(zip(sel_agg_label, cond_op_label)):
            if agg_op < len(SQL.agg_sql_dict):
                sel.append(col_id)
                agg.append(int(agg_op))
            if cond_op < len(SQL.op_sql_dict):
                conds.append([col_id, int(cond_op)])
        return {
            'sel': sel,
            'agg': agg,
            'cond_conn_op': cond_conn_op,
            'conds': conds
        }

class CustomDataset(Dataset):

    # customized dataset
    # implement ___len___ & __getitem__ function

    def __init__(
        self, 
        data, 
        max_len=config.MAX_LEN, 
        max_headers=config.MAX_HEADERS,
        sql_label_encoder=SqlLabelEncoder(),
        model_name=config.BASE_MODEL_PATH, 
        SEP_temp='|', 
        REAL_temp='?',
        TEXT_temp='!',
        cls_token='[unused8]', 
        REAL_token='[unused20]', 
        TEXT_token='[unused21]'
    ):
        self.data        = data                                                              # loaded data
        self.max_len     = max_len                                                           # max length of sequence
        self.max_headers = max_headers                                                       # max length of count of headers
        self.tokenizer   = BertTokenizer.from_pretrained(model_name, cls_token=cls_token)    # cls_token will be replaced by a unused token in bert vocab 
        self.indexes     = np.arange(len(self.data))                                         # set a list of indexes according to data length
        self.sql_label_encoder = sql_label_encoder                                           # label encoder for SQL objects

        self.CLS_id  = self.tokenizer.encode([cls_token])[1]                                 # CLS's token id 
        self.REAL_id = self.tokenizer.encode([REAL_token])[1]                                # REAL's token id 
        self.TEXT_id = self.tokenizer.encode([TEXT_token])[1]                                # TEXT's token id 
        self.SEP_id  = self.tokenizer.encode('[SEP]')[1]                                     # SEP's token id 

        self.SEP_temp  = SEP_temp                                                            # temporarily symbol for SEP token
        self.REAL_temp = REAL_temp                                                           # temporarily symbol for REAL token
        self.TEXT_temp = TEXT_temp                                                           # temporarily symbol for TEXT token      

        self.SEP_temp_id  = self.tokenizer.encode(SEP_temp)[1]                               # SEP_temp's token id 
        self.REAL_temp_id = self.tokenizer.encode(REAL_temp)[1]                              # REAL_temp's token id 
        self.TEXT_temp_id = self.tokenizer.encode(TEXT_temp)[1]                              # TEXT_temp's token id 


    def __len__(self):  
        return len(self.data) 

    def __getitem__(self, idx): 
        
        question = str(self.data[idx].question)

        # construct header in str
        header_sent = ''
        for header, label in self.data[idx].table.header:
            if label == 'text':
                header_sent += self.TEXT_temp
            if label == 'real':
                header_sent += self.REAL_temp
            header_sent += re.sub(r'[\(\（].*[\)\）]', '', header)   # remove brackets and its content for headers
            header_sent += self.SEP_temp
        header_sent = header_sent[:-1]

        embeddings = self.tokenizer(
            question, header_sent,                    # sentence 1, sentence 2
            padding='max_length',                     # Pad to max_length
            truncation=True,                          # Truncate to max_length
            max_length=self.max_len,                  # Set max_length
            return_tensors='pt'                       # Return torch.Tensor objects
        )

        token_ids = torch.squeeze(embeddings['input_ids'])                                               # tensor of token ids
        token_ids = torch.where(token_ids==self.SEP_temp_id,  torch.tensor(self.SEP_id), token_ids)      # replace SEP_temp_id by SEP_id
        token_ids = torch.where(token_ids==self.REAL_temp_id, torch.tensor(self.REAL_id), token_ids)     # replace REAL_temp_id by REAL_id
        token_ids = torch.where(token_ids==self.TEXT_temp_id, torch.tensor(self.TEXT_id), token_ids)     # replace TEXT_temp_id by TEXT_id

        attention_masks = torch.squeeze(embeddings['attention_mask'])                                    # binary tensor with "0" for padded values and "1" for the other values

        header_ids = [i for i, value in enumerate(token_ids) if value == self.REAL_id or value == self.TEXT_id]    # list of SEP positions, length: nb of cols + 1
        header_ids_padded = header_ids + [0] * (self.max_headers - len(header_ids))                                # padding
        header_masks = [1] * len(header_ids) + [0] * (self.max_headers - len(header_ids))                          # binary tensor with "0" for padded values and "1" for the other values

        # True if the dataset has labels (when training or validating or testing)
        if self.sql_label_encoder is not None:       
            COND_CONN_OP, SEL_AGG, COND_OP = self.sql_label_encoder.encode(self.data[idx].sql, num_cols=len(self.data[idx].table.header))             
            SEL_AGG = SEL_AGG.tolist() + [0] * (self.max_headers - len(SEL_AGG))   # padding
            COND_OP = COND_OP.tolist() + [0] * (self.max_headers - len(COND_OP))   # padding
            return {
                ### X
                'token_ids'       : token_ids,
                'attention_masks' : attention_masks,
                'header_ids'      : torch.tensor(header_ids_padded),
                'header_masks'    : torch.tensor(header_masks),
                ### y
                'COND_CONN_OP'    : torch.tensor(COND_CONN_OP),
                'SEL_AGG'         : torch.tensor(SEL_AGG),
                'COND_OP'         : torch.tensor(COND_OP)         
            } 
        # False if the dataset do not have labels (when inferencing)
        else:                                                           
            return {
                'token_ids'       : token_ids,
                'attention_masks' : attention_masks,
                'header_ids'      : torch.tensor(header_ids_padded),
                'header_masks'    : torch.tensor(header_masks),
            }
            
