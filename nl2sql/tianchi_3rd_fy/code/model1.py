"""
Created by FENG YUAN on 2021/10/6
"""

import torch
from transformers import BertConfig, BertModel
from transformers import AlbertConfig, AlbertModel

import config


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.bert_config = AlbertConfig.from_pretrained(config.BASE_MODEL_PATH, output_hidden_states=True, return_dict=True)
        self.bert_as_encoder = AlbertModel.from_pretrained(config.BASE_MODEL_PATH, config=self.bert_config)
        self.dense_COND_CONN_OP = torch.nn.Linear(self.bert_config.hidden_size, len(config.COND_CONN_OP_DICT))
        self.dense_SEL_AGG = torch.nn.Linear(self.bert_config.hidden_size, len(config.SEL_AGG_DICT) + 1)  # +1 to add 'NO_OP'
        self.dense_COND_OP = torch.nn.Linear(self.bert_config.hidden_size, len(config.COND_OP_DICT) + 1)  # +1 to add 'NO_OP'

    def forward(self, token_ids, attention_masks, header_ids):

        last_hidden_state = self.bert_as_encoder(token_ids, attention_masks).last_hidden_state       
        # the last hidden state of all tokens of BERT in shape (batch_size, max_len, hidden_states)

        ### Subtask1: COND_CONN_OP
        h_CLS = last_hidden_state[:, 0, :] 
        # the CLS token's representation in shape (batch_size, hidden_states)
        COND_CONN_OP_output = self.dense_COND_CONN_OP(h_CLS)
        # logits in shape (batch_size, COND_CONN_OP classes)

        header_ids = header_ids.unsqueeze(-1).expand(-1, -1, self.bert_config.hidden_size).to(dtype=torch.int64)
        # (batch_size, h_headers_len) -> (batch_size, h_headers_len, hidden_states)
        h_headers = torch.gather(input=last_hidden_state, dim=1, index=header_ids)
        # equivalent to tf.batch_gather, in shape (batch_size, h_headers_len, hidden_states)

        ### Subtask2: SEL_AGG
        SEL_AGG_output = self.dense_SEL_AGG(h_headers) 
        # in shape (batch_size, h_headers_len, nb_classes)

        ### Subtask3: COND_OP
        COND_OP_output = self.dense_COND_OP(h_headers)  
        # in shape (batch_size, h_headers_len, nb_classes)

        return {
            'COND_CONN_OP': COND_CONN_OP_output,
            'SEL_AGG': SEL_AGG_output,
            'COND_OP': COND_OP_output
        }
