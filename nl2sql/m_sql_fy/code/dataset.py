
import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset

import config


class CustomDataset(Dataset):

    # customized dataset
    # implement ___len___ & __getitem__ function

    def __init__(self, data, has_label=True, max_len=config.MAX_LEN, model_name=config.BASE_MODEL_PATH, SEP_temp='|', cls_token='[unused8]'):
        self.data = data                                                                     # loaded data
        self.max_len = max_len                                                               # max length of sequence
        self.tokenizer = BertTokenizer.from_pretrained(model_name, cls_token=cls_token)      # cls_token will be replaced by a unused token in bert vocab
        self.indexes = np.arange(len(self.data))                                             # set a list of indexes according to data length
        self.has_label = has_label                                                           # bool, input data contains labels or not
        self.SEP_temp = SEP_temp                                                             # temporarily seperate headers, len(SEP_temp) should be 1, then will be replaced by [SEP]'s token id 
        self.SEP_temp_id = self.tokenizer.encode(SEP_temp)[1]                                # SEP_temp's token id 
        self.SEP_id = self.tokenizer.encode('[SEP]')[1]                                      # SEP's token id 
        self.XLS_id = self.tokenizer.encode(cls_token)[1]                                    # XLS's token id 

    def __len__(self):  
        return len(self.data) 

    def __getitem__(self, idx): 
        
        question = str(self.data[idx].question)

        # construct header in str
        header_sent = ''
        for header, _ in self.data[idx].table.header:
            header_sent+=header
            header_sent+=self.SEP_temp
        header_sent = header_sent[: -1]               # drop the SEP_temp at tail

        # print(question)
        # print(header_sent)

        embeddings = self.tokenizer(
            question, header_sent,                    # sentence 1, sentence 2
            padding='max_length',                     # Pad to max_length
            truncation=True,                          # Truncate to max_length
            max_length=self.max_len,                  # Set max_length
            return_tensors='pt'                       # Return torch.Tensor objects
        )

        token_ids = torch.squeeze(embeddings['input_ids'])                                             # tensor of token ids
        token_ids = torch.where(token_ids==self.SEP_temp_id, torch.tensor(self.SEP_id), token_ids)     # replace SEP_temp_id by SEP_id

        attention_masks = torch.squeeze(embeddings['attention_mask'])    # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = torch.squeeze(embeddings['token_type_ids'])     # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        SEP_ids = [i for i, value in enumerate(token_ids) if value == self.SEP_id]   # list of SEP positions, length: nb of cols + 1

        # question_masks: 1 for question tokens (including XLS and 1st SEP), 0 for others  
        question_len = len(question) + 2
        ones = torch.ones(question_len)
        zeros = torch.zeros(self.max_len - question_len)
        question_masks = torch.cat([ones, zeros])

        # True if the dataset has labels (when training or validating or testing)
        if self.has_label:                                               
            W_num_op_label = self.data[idx].sql.conn_sql_dict[self.data[idx].sql.cond_conn_op] + '-' + str(len(self.data[idx].sql.conds)) 
            S_num_label = len(self.data[idx].sql.sel)
            return {
                ### X
                'token_ids': token_ids,
                'token_type_ids': token_type_ids,
                'attention_masks': attention_masks,
                'SEP_ids': torch.tensor(SEP_ids),
                'SEP_masks': torch.ones(len(SEP_ids)),
                'question_masks': question_masks,
                ### y
                'S_num': torch.tensor(config.S_num_label2id[S_num_label]),              
                'W_num_op': torch.tensor(config.W_num_op_label2id[W_num_op_label]),
            } 
        # False if the dataset do not have labels (when inferencing)
        else:                                                           
            return {
                'token_ids': token_ids,
                'token_type_ids': token_type_ids,
                'attention_masks': attention_masks,
                'SEP_ids': torch.tensor(SEP_ids),
                'SEP_masks': torch.ones(len(SEP_ids)),
                'question_masks': question_masks,
            }
            

def collate_fn(batch_data):
    '''
    SEP_ids lengths are different in each batch, need padding
    '''
    batch_data.sort(key=lambda xi: len(xi['SEP_ids']), reverse=True)
    SEP_ids_seq = [xi['SEP_ids'] for xi in batch_data]
    padded_SEP_ids_seq = torch.nn.utils.rnn.pad_sequence(SEP_ids_seq, batch_first=True, padding_value=0)

    SEP_masks_seq = [xi['SEP_masks'] for xi in batch_data]
    padded_SEP_masks_seq = torch.nn.utils.rnn.pad_sequence(SEP_masks_seq, batch_first=True, padding_value=0)

    token_ids_seq = [xi['token_ids'] for xi in batch_data]
    token_type_ids_seq = [xi['token_type_ids'] for xi in batch_data]
    attention_masks_seq = [xi['attention_masks'] for xi in batch_data]
    question_masks_seq = [xi['question_masks'] for xi in batch_data]
    S_num_seq = [xi['S_num'] for xi in batch_data]
    W_num_op_seq = [xi['W_num_op'] for xi in batch_data]

    return {
        ### X
        'token_ids': torch.stack(token_ids_seq) , 
        'token_type_ids': torch.stack(token_type_ids_seq),
        'attention_masks': torch.stack(attention_masks_seq),
        'question_masks': torch.stack(question_masks_seq),
        'SEP_ids': padded_SEP_ids_seq,
        'SEP_masks': padded_SEP_masks_seq,
        ### y
        'S_num': torch.stack(S_num_seq),
        'W_num_op': torch.stack(W_num_op_seq)
    }


def collate_fn_labelless(batch_data):
    '''
    SEP_ids lengths are different in each batch, need padding
    For no label case
    '''
    batch_data.sort(key=lambda xi: len(xi['SEP_ids']), reverse=True)
    SEP_ids_seq = [xi['SEP_ids'] for xi in batch_data]
    padded_SEP_ids_seq = torch.nn.utils.rnn.pad_sequence(SEP_ids_seq, batch_first=True, padding_value=0)

    SEP_masks_seq = [xi['SEP_masks'] for xi in batch_data]
    padded_SEP_masks_seq = torch.nn.utils.rnn.pad_sequence(SEP_masks_seq, batch_first=True, padding_value=0)

    token_ids_seq = [xi['token_ids'] for xi in batch_data]
    token_type_ids_seq = [xi['token_type_ids'] for xi in batch_data]
    attention_masks_seq = [xi['attention_masks'] for xi in batch_data]
    question_masks_seq = [xi['question_masks'] for xi in batch_data]

    return {
        ### X
        'token_ids': torch.stack(token_ids_seq) , 
        'token_type_ids': torch.stack(token_type_ids_seq),
        'attention_masks': torch.stack(attention_masks_seq),
        'question_masks': torch.stack(question_masks_seq),
        'SEP_ids': padded_SEP_ids_seq,
        'SEP_masks': padded_SEP_masks_seq,
    }