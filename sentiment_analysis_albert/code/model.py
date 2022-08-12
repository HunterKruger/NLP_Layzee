import torch
from transformers import AlbertConfig, AlbertModel
import config

class Albert(torch.nn.Module):

    def __init__(self) :
        super(Albert,self).__init__()
        self.albert_config = AlbertConfig.from_pretrained(config.BASE_MODEL_PATH, output_hidden_states=True,return_dict=True)
        self.albert_as_encoder = AlbertModel.from_pretrained(config.BASE_MODEL_PATH, config=self.albert_config)
        self.dense = torch.nn.Linear(self.albert_config.hidden_size * len(config.HIDDEN_STATE_LIST), config.DENSE_UNITS)
        self.dense2= torch.nn.Linear(config.DENSE_UNITS, config.CLASSES)
        self.pooling = torch.nn.AdaptiveMaxPool1d(1)
        self.hidden_state_list = config.HIDDEN_STATE_LIST
        self.dropout = torch.nn.Dropout(p=config.DROPOUT_RATE)
        self.relu = torch.nn.ReLU()
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        out = self.albert_as_encoder(input_ids, token_type_ids, attention_mask)
        temp_list = []
        for layer_index in self.hidden_state_list:  # not useful for Albert because params are shared between layers
            temp = out.hidden_states[layer_index]   # (batch_size/num_gpus, max_len, hidden_state)
            temp = temp.permute(0,2,1)              # (batch_size/num_gpus, hidden_state, max_len)
            temp = self.pooling(temp)               # (batch_size/num_gpus, hidden_state, 1)
            temp = torch.squeeze(temp)              # (batch_size/num_gpus, hidden_state)
            temp_list.append(temp)
        out = torch.cat(temp_list, dim=-1)          # (batch_size/num_gpus, hidden_state*len_hidden_state_list)
        out = self.dropout(out)
        out = self.dense(out)   
        out = self.relu(out) 
        out = self.dropout(out)
        out = self.dense2(out)    
        return out
