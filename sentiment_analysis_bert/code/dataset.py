import re
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from transformers import BertTokenizer
import config
from sklearn.utils import shuffle as reset


def process_data(filename=config.INPUT_FILE, classes2idx=config.CLS2IDX, with_labels=True):
    """
    Read csv data then map labels.
    classes2idx = key is label name, value is label number
    filename->str: file path
    with_labels->bool: map or not map
    return: a dataframe
    """
    df = pd.read_csv(filename, encoding='utf-8')
    if with_labels:
        df = df.replace({'class_label': classes2idx})  # mapping
    return df


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

    train = data_df[int(len(data_df) * test_size):].reset_index(drop=True)
    test = data_df[:int(len(data_df) * test_size)].reset_index(drop=True)

    return train, test


class CustomDataset(tf.keras.utils.Sequence):

    # customized dataset
    # implement ___len___ & __getitem__ function

    def __init__(self, 
                 sentences, 
                 batch_size=config.BATCH_SIZE,
                 labels=None,
                 max_len=config.MAX_LEN, 
                 model_name=config.BASE_MODEL_PATH,
                 shuffle=True
                ):

        self.sentences = sentences 
        self.batch_size = batch_size
        self.max_len = max_len  # max length of sequence
        self.tokenizer = BertTokenizer.from_pretrained(model_name)  # Initialize the tokenizer from model path in HuggingFace style
        self.indexes = np.arange(len(self.sentences))
        self.labels = labels
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):  
        return int(np.ceil(len(self.sentences) / self.batch_size))
    
    def get_size(self):
        return len(self.sentences) 
    
    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
             np.random.RandomState(config.RANDOM_STATE).shuffle(self.indexes)

    def __getitem__(self, idx): 
        
        # Retrieves the batch of index.
        if len(self.sentences) == 1:
            indexes = 0 
            sents = self.sentences[indexes]
            encoded_sents = self.tokenizer(
                sents,
                padding='max_length',     # Pad to max_length
                truncation=True,          # Truncate to max_length
                max_length=self.max_len,  # Set max_length
                return_tensors='tf'       # Return tf.Tensor objects
            )
        else:
            indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
            # Selecting a sentence at the specified index in the data frame
            sents = self.sentences[indexes]
            # Tokenize the pair of sentences to get token ids, attention masks and token type ids
            encoded_sents = self.tokenizer.batch_encode_plus(
                sents.tolist(),
                padding='max_length',     # Pad to max_length
                truncation=True,          # Truncate to max_length
                max_length=self.max_len,  # Set max_length
                return_tensors='tf'       # Return tf.Tensor objects
            )


        token_ids = tf.squeeze(encoded_sents['input_ids'])            # tensor of token ids
        attention_masks = tf.squeeze(encoded_sents['attention_mask']) # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = tf.squeeze(encoded_sents['token_type_ids'])  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.labels is not None:  # True if the dataset has labels (when training or validating)
            labels = tf.squeeze(self.labels[indexes])
            return [token_ids, attention_masks, token_type_ids], labels
        else:  # when inferencing
            return [token_ids, attention_masks, token_type_ids]
