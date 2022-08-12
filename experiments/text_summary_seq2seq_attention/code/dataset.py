import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
from config import config
import jieba
import io
import json
    

def create_dataset(
    contexts, 
    titles, 
    fit_transform=False,
    reverse=config.REVERSE,
    tokenizer_path=config.TOKENIZER_PATH, 
    max_len_encoder=config.MAX_LEN_ENCODER, 
    max_len_decoder=config.MAX_LEN_DECODER,
    num_words=config.NUM_WORDS
):
 
    contexts = list(map(lambda x: jieba.lcut(x), contexts))                   # segmentation
    contexts = list(map(lambda x: ['<START>'] + x + ['<END>'], contexts))     # add special tokens
   
    if titles is not None:
        titles = list(map(lambda x: jieba.lcut(x), titles))                # segmentation
        titles_in = list(map(lambda x: ['<START>'] + x , titles))          # add special tokens
        titles_out = list(map(lambda x: x + ['<END>'], titles))            # add special tokens

        
    if fit_transform:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token='<UNK>', lower=False)  
        tokenizer.fit_on_texts(contexts + titles) 
        ## Save tokenizer
        tokenizer_json = tokenizer.to_json()
        with io.open(tokenizer_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    else:
        ## Read tokenizer
        with open(tokenizer_path) as f:
            data = json.load(f)
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
    
    X = tokenizer.texts_to_sequences(contexts)               
    X = tf.keras.preprocessing.sequence.pad_sequences(X, value=0, padding='post', maxlen=max_len_encoder)
    if reverse:            # reverse, for better performance 
        X = X[:,::-1] 
            
    if titles is None:
        return X, tokenizer
    else:    
        y_in = tokenizer.texts_to_sequences(titles_in)               
        y_in = tf.keras.preprocessing.sequence.pad_sequences(y_in, value=0, padding='post', maxlen=max_len_decoder)     # set <PAD>=0
        y_out = tokenizer.texts_to_sequences(titles_out)               
        y_out = tf.keras.preprocessing.sequence.pad_sequences(y_out, value=0, padding='post', maxlen=max_len_decoder)   # set <PAD>=0
        return X, y_in, y_out, tokenizer

    
                              
                
