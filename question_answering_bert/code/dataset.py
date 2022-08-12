import re
import pandas as pd
import numpy as np
import tensorflow as tf

from pathlib import Path
from transformers import BertTokenizer
import config


class CustomDataset(tf.keras.utils.Sequence):

    # customized dataset
    # implement ___len___ & __getitem__ function

    def __init__(self, 
                 questions, 
                 documents, 
                 starts=None, 
                 ends=None,
                 max_len=config.MAX_LEN, 
                 model_name=config.BASE_MODEL_PATH, 
                 with_labels=True,
                 batch_size=config.BATCH_SIZE
                ):

        self.questions = questions 
        self.documents = documents               
        self.starts = starts               
        self.ends = ends               
        self.tokenizer = BertTokenizer.from_pretrained(model_name)  # Initialize the tokenizer from model path in HuggingFace style
        self.max_len = max_len          # max length of sequence
        self.with_labels = with_labels  # data with labels or not
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.questions))

    def __len__(self):  # return batch count
        return int(np.ceil(len(self.questions)/self.batch_size))
    
    def get_sample_count(self):
        return len(self.questions) 

    def __getitem__(self, idx):  # get tokenized sample by idx
        
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        # Selecting a sentence at the specified idx in the data frame
        documents = self.documents[indexes]
        questions = self.questions[indexes]
        documents = [[letter for letter in document] for document in documents]
        questions = [[letter for letter in question] for question in questions]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(documents, questions,
                                      is_split_into_words=True,   # sequence has already in list of str
                                      padding='max_length',       # Pad to max_length
                                      truncation=True,            # Truncate to max_length
                                      max_length=self.max_len,    # Set max_length
                                      return_tensors='tf')        # Return tf.Tensor objects

        token_ids = tf.squeeze(encoded_pair['input_ids'])            # tensor of token ids
        attention_masks = tf.squeeze(encoded_pair['attention_mask']) # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = tf.squeeze(encoded_pair['token_type_ids'])  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens
        
        if not self.with_labels:  # True if the dataset has labels (when training or validating)
            return [token_ids, attention_masks, token_type_ids]
        else:  # when inferencing
            start_ids = self.starts[indexes]
            end_ids = self.ends[indexes]
            start_ids += 1    # offset calibration
            end_ids += 1      # offset calibration
            start_ids = tf.squeeze(start_ids)
            end_ids = tf.squeeze(end_ids)
            return [token_ids, attention_masks, token_type_ids], [start_ids, end_ids]

    
