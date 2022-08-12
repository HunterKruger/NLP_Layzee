import re
import numpy as np
import tensorflow as tf
from pathlib import Path
from transformers import BertTokenizer
import config


class CustomDataset(tf.keras.utils.Sequence): 
    
    # customized dataset
    # implement ___len___ & __getitem__ function
    
    def __init__(self, 
                 data, 
                 batch_size=config.BATCH_SIZE, 
                 maxlen=config.MAX_LEN, 
                 model_name=config.BASE_MODEL_PATH,
                 with_labels=True
                ):

        self.data = data                                                         # filepath, with filename
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained(model_name)               # Initialize the tokenizer from model path in HuggingFace style
        self.maxlen = maxlen                                                     # max length of sequence
        self.with_labels = with_labels                                           # data with labels or not
        if self.with_labels:
            # token_docs = [['a','b','c'],['a','b','c'],...]
            # tag_docs_in_digit = [[1, 2, 3], [1, 2, 3], ...]
            self.token_docs, self.tag_docs_in_digit = self._read_data_with_labels()
        else:
            self.token_docs = self._read_data_without_labels()
        self.indexes = np.arange(len(self.token_docs))
                
    def _read_data_with_labels(self):
        # read data from 'data'
        file_path = Path(self.data)
        raw_text = file_path.read_text().strip()
        raw_docs = re.split(r'\n\t?\n', raw_text)
        token_docs = []
        tag_docs = []
        # print(raw_docs[1])
        for doc in raw_docs:
            tokens = []
            tags = []
            # print(doc)
            for line in doc.split('\n'):
                # print(line)
                token, tag = line.split(' ')
                tokens.append(token)
                tags.append(tag)
            token_docs.append(tokens)
            tag_docs.append(tags)
        tag_docs_in_id = [[config.TAG2ID[tag] for tag in tags] for tags in tag_docs]
        return token_docs, tag_docs_in_id

    def _read_data_without_labels(self):
        file_path = Path(self.data)
        raw_text = file_path.read_text().strip()
        raw_docs = re.split(r'\n\t?\n', raw_text)
        token_docs = []
        # print(raw_docs[1])
        for doc in raw_docs:
            tokens = []
            for line in doc.split('\n'):
                token = line
                tokens.append(token)
            token_docs.append(tokens)
        return token_docs
    

    def __len__(self):  # return batch count
        return int(np.ceil((len(self.token_docs) / self.batch_size)))
    
    def get_sample_count(self):
        return len(self.token_docs) 
    
    def __getitem__(self, idx): # get tokenized sample by index
        
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        # Selecting a sentence at the specified index in the self.token_docs
        sents = [self.token_docs[x] for x in indexes]
        
        # Tokenize the sentence to get token ids, attention masks and token type ids
        encoded_sents = self.tokenizer.batch_encode_plus(sents,
                                       is_split_into_words=True,    # sent has already in list of str
                                       return_offsets_mapping=False, 
                                       padding='max_length',        # Pad to max_length
                                       truncation=True,             # Truncate to max_length
                                       max_length=config.MAX_LEN,   # Set max_length
                                       return_tensors='tf')         # Return tf.Tensor objects
        
        # tensor of token ids
        token_ids = tf.squeeze(encoded_sents['input_ids'])          
        
        # binary tensor with "0" for padded values and "1" for the other values
        attn_masks = tf.squeeze(encoded_sents['attention_mask'])    
        
        # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens
        token_type_ids = tf.squeeze(encoded_sents['token_type_ids'])  
        
        if not self.with_labels:  # if the dataset does not have labels (usually when testing)
            return [token_ids, attn_masks, token_type_ids]  
        else:                     # if the dataset has labels (usually when training or validating)
            # tokenize tag id sequence manually
            tags = [self.tag_docs_in_digit[x] for x in indexes]
            label_ids = []
            for t in tags:
                if len(t) > (config.MAX_LEN-2):
                    t = t[:(config.MAX_LEN-2)]
                l = t.copy()
                l.insert(0, config.TAG2ID['CLS'])
                l.append(config.TAG2ID['SEP'])
                l.extend([config.TAG2ID['PAD'] for x in range(config.MAX_LEN-len(t)-2)])
                label_ids.append(l)
            return [token_ids,  attn_masks, token_type_ids], tf.convert_to_tensor(label_ids)  