import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from transformers import  BertTokenizer
import config
from sklearn.utils import shuffle as reset
from imblearn.under_sampling import RandomUnderSampler


def undersampling(df):
    rus = RandomUnderSampler(sampling_strategy='majority')
    X = df.drop([config.LABEL_FIELD], axis=1)
    y = df[config.LABEL_FIELD]
    X_res, y_res = rus.fit_resample(X, y)
    df_ = pd.concat([X_res, y_res], axis=1)
    df_.drop_duplicates(inplace=True)    
    df_ = df_.sample(frac=1)       
    df_ = df_.reset_index()
    df_.drop('index',axis=1,inplace=True)
    return df_


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
 
        if self.labels is None:  
            return [token_ids, attention_masks, token_type_ids]
        else:  
            labels = tf.squeeze(self.labels[indexes])
            return [token_ids, attention_masks, token_type_ids], labels



class CustomDatasetSiamese(tf.keras.utils.Sequence): 
    
    # customized dataset
    # implement ___len___ & __getitem__ function
    
    def __init__(self, 
                 sent, sent2,
                 batch_size=config.BATCH_SIZE,
                 labels=None,
                 max_len=config.MAX_LEN_SIAMESE, 
                 model_name=config.BASE_MODEL_PATH, 
                ):
        self.sent = sent
        self.sent2 = sent2
        self.batch_size = batch_size
        self.max_len = max_len                                      # max length of sequence
        self.tokenizer = BertTokenizer.from_pretrained(model_name)  # Initialize the tokenizer from model path in HuggingFace style
        self.indexes = np.arange(len(self.sent))
        self.labels = labels

    def __len__(self):  # return sample count
        return int(np.ceil(len(self.sent) / self.batch_size))
    
    def get_counts(self):
        return len(self.sent)

    def __getitem__(self, idx): # get tokenized sample by index

        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
 
        # Selecting a sentence at the specified index in the data frame
        sents = self.sent[indexes]
        sents2 = self.sent2[indexes]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_sents = self.tokenizer.batch_encode_plus(
            sents.tolist(), 
            padding='max_length',     # Pad to max_length
            truncation=True,          # Truncate to max_length
            max_length=self.max_len,  # Set max_length
            return_tensors='tf'       # Return tf.Tensor objects
        )

        token_ids = tf.squeeze(encoded_sents['input_ids'])               # tensor of token ids
        attention_masks = tf.squeeze(encoded_sents['attention_mask'])    # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = tf.squeeze(encoded_sents['token_type_ids'])     # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens
 
        encoded_sents2 = self.tokenizer.batch_encode_plus(
            sents2.tolist(), 
            padding='max_length',     # Pad to max_length
            truncation=True,          # Truncate to max_length
            max_length=self.max_len,  # Set max_length
            return_tensors='tf'       # Return tf.Tensor objects
        )

        token_ids2 = tf.squeeze(encoded_sents2['input_ids'])               # tensor of token ids
        attention_masks2 = tf.squeeze(encoded_sents2['attention_mask'])    # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids2 = tf.squeeze(encoded_sents2['token_type_ids'])     # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens
       
        if self.labels is None: 
            return [[token_ids, attention_masks, token_type_ids], [token_ids2, attention_masks2, token_type_ids2]]
        else:  
            labels = tf.squeeze(self.labels[indexes])
            return [[token_ids, attention_masks, token_type_ids], [token_ids2, attention_masks2, token_type_ids2]], labels        
