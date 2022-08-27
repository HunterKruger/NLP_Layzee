import re
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from transformers import BertTokenizer
import config
from sklearn.utils import shuffle as reset


def split_df(df, test_ratio=0.2, val_ratio=None, target=None, random_state=1337):
    """
    Split a dataset into training set and test set
    df -> (train, test)
       -> (X_train, X_test, y_train, y_test)
    :param df: a DataFrame to be split
    :param test_ratio: ratio of test set, 0-1
    :param val_ratio: ratio of validation set, 0-1
        split into (train, test) if not specified
        split into (train, val, test) if specified
    :param target:
        split into (train, test) if not specified
        split into (X_train, X_test, y_train, y_test) if specified
    :param random_state: random state
    """
    if target:
        if val_ratio:
            count = df.shape[0]
            val_count = int(count * val_ratio)
            test_count = int(count * test_ratio)
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            val = df[:val_count]
            test = df[val_count:(val_count + test_count)]
            train = df[(val_count + test_count):]
            val = val.sample(frac=1, random_state=random_state).reset_index(drop=True)
            test = test.sample(frac=1, random_state=random_state).reset_index(drop=True)
            train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)
            X_train = train.drop(target, axis=1, inplace=False)
            X_val = val.drop(target, axis=1, inplace=False)
            X_test = test.drop(target, axis=1, inplace=False)
            y_train = train[target]
            y_val = val[target]
            y_test = test[target]
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            X = df.drop(target, axis=1, inplace=False)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)
            return X_train, X_test, y_train, y_test
    else:
        if val_ratio:
            count = df.shape[0]
            val_count = int(count * val_ratio)
            test_count = int(count * test_ratio)
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            val = df[:val_count]
            test = df[val_count:(val_count + test_count)]
            train = df[(val_count + test_count):]
            val = val.sample(frac=1, random_state=random_state).reset_index(drop=True)
            test = test.sample(frac=1, random_state=random_state).reset_index(drop=True)
            train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)
            return train, val, test
        else:
            train, test = train_test_split(df, test_size=test_ratio, random_state=random_state)
            return train, test

class CustomDataset(tf.keras.utils.Sequence):

    # customized dataset
    # implement ___len___ & __getitem__ function

    def __init__(self, 
                 sentences, sentences2=None,
                 batch_size=config.BATCH_SIZE,
                 labels=None,
                 max_len=config.MAX_LEN, 
                 model_name=config.BASE_MODEL_PATH,
                 shuffle=True
                ):

        self.sentences = sentences 
        self.sentences2 = sentences2 
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
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        if self.sentences2 is None:
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
        else:
            # Selecting a sentence at the specified index in the data frame
            sents = self.sentences[indexes]
            sents2 = self.sentences2[indexes]
            # Tokenize the pair of sentences to get token ids, attention masks and token type ids
            encoded_sents = self.tokenizer.batch_encode_plus(
                list(zip(sents, sents2)),
                padding='max_length',     # Pad to max_length
                truncation='only_second', # Truncate to max_length
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
