import re
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from transformers import AutoTokenizer, BertTokenizer
from config import config
from sklearn.utils import shuffle as reset


def train_test_split(data_df, test_size=0.2, shuffle=True, random_state=None):
    """
    Split dataframe into training set and test set.
    data_df->DataFrame: to be splited
    test_size->float: proportion for test set
    shuffle->bool: shuffle before split
    random_state->int: random seed 
    return: training set & test set
    """
    if shuffle:
        data_df = reset(data_df, random_state=random_state)
        
    train = data_df[int(len(data_df)*test_size):].reset_index(drop = True)
    test  = data_df[:int(len(data_df)*test_size)].reset_index(drop = True)

    return train, test


class CustomDataset(tf.keras.utils.Sequence): 
    
    # customized dataset
    # implement ___len___ & __getitem__ function
    
    def __init__(self, 
                 sentence_pairs,
                 batch_size=config.BATCH_SIZE,
                 labels=None,
                 max_len=config.MAX_LEN, 
                 model_name=config.BASE_MODEL_PATH, 
                ):

        self.sentence_pairs = sentence_pairs
        self.batch_size = batch_size
        self.max_len = max_len                                      # max length of sequence
        self.tokenizer = BertTokenizer.from_pretrained(model_name)  # Initialize the tokenizer from model path in HuggingFace style
        self.indexes = np.arange(len(self.sentence_pairs))
        self.labels = labels

    def __len__(self):  # return sample count
        return int(np.ceil(len(self.sentence_pairs) / self.batch_size))
    
    def get_counts(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx): # get tokenized sample by index

        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
 
        # Selecting a sentence at the specified index in the data frame
        sent_pairs = self.sentence_pairs[indexes]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer.batch_encode_plus(
                                      sent_pairs.tolist(), 
                                      padding='max_length',     # Pad to max_length
                                      truncation=True,          # Truncate to max_length
                                      max_length=self.max_len,  # Set max_length
                                      return_tensors='tf')      # Return tf.Tensor objects
        
        token_ids = tf.squeeze(encoded_pair['input_ids'])               # tensor of token ids
        attention_masks = tf.squeeze(encoded_pair['attention_mask'])    # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = tf.squeeze(encoded_pair['token_type_ids'])     # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens
 
        if self.labels is not None:  # True if the dataset has labels (when training or validating)
            labels = tf.squeeze(self.labels[indexes])
            return [token_ids, attention_masks, token_type_ids], labels
        else:  # when inferencing
            return [token_ids, attention_masks, token_type_ids]  