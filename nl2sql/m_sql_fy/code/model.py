import torch
from transformers import BertConfig, BertModel

import config


class MSQL(torch.nn.Module):

    def __init__(self) :

        super(MSQL,self).__init__()
        self.bert_config = BertConfig.from_pretrained(config.BASE_MODEL_PATH, output_hidden_states=True,return_dict=True)
        self.bert_as_encoder = BertModel.from_pretrained(config.BASE_MODEL_PATH, config=self.bert_config)
        self.dense_S_num = torch.nn.Linear(self.bert_config.hidden_size, len(config.S_num_labels))
        self.dense_W_num_op = torch.nn.Linear(self.bert_config.hidden_size, len(config.W_num_op_labels))
        self.dense_W_col_val = torch.nn.Linear(self.bert_config.hidden_size, len(config.W_col_val_labels))


    def forward(self, token_ids, token_type_ids, attention_masks, SEP_ids, SEP_masks, question_masks):

        last_hidden_state = self.bert_as_encoder(token_ids, token_type_ids, attention_masks).last_hidden_state     # out: the last hidden state of all tokens of BERT, shape (batch_size, max_len, hidden_states)

        h_XLS = last_hidden_state[:,0,:]                # out: the XLS token's representation, shape (batch_size, hidden_states)

        ### Subtask1: S_num
        S_num_output = self.dense_S_num(h_XLS)          # out: logits in shape (batch_size, S_num classes)

        ### Subtask2: W_num_op
        W_num_op_output = self.dense_W_num_op(h_XLS)    # out: logits in shape (batch_size, W_num_op classes)

        # print(SEP_ids.shape)                                  # batch_size, nb of SEPs (maximum in this batch)
        # first_SEP_position = SEP_ids[:,0]                     # batch_size
        # first_SEP_position = first_SEP_position[:,None]       # batch_size, 1

        ### Subtask3: W_col_val
        W_col_val_output = self.dense_W_col_val(last_hidden_state)                           # out: logits for all tokens, in shape (batch_size, max_len, W_col_val classes)
        question_masks = question_masks[:,:,None]                                            # out: shape (batch_size, max_len, 1)
        question_masks = question_masks.expand((-1, -1, len(config.W_col_val_labels)))       # out: shape (batch_size, max_len, W_col_val classes)
        W_col_val_output = torch.where(question_masks==1, W_col_val_output, question_masks)  # out: logits for all tokens with PAD positions set to 0, in shape (batch_size, max_len, W_col_val classes)

        # get h_c: representations of all columns

        return {
            'S_num': S_num_output,
            'W_num_op': W_num_op_output,
            'W_col_val': W_col_val_output
        }
        

