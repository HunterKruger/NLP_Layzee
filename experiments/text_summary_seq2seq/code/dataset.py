import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
from config import config
import jieba
import io
import json

def load_embedding(path=config.WORD2VEC_FILE):
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
    length = len(word2vec[0])
    rng_pad = np.random.RandomState(0)
    rng_oov = np.random.RandomState(1)
    rng_start = np.random.RandomState(2)
    rng_end = np.random.RandomState(3)
    vec_pad = (rng_pad.rand(length) - 0.5) * 2
    vec_oov = (rng_oov.rand(length) - 0.5) * 2
    vec_start = (rng_start.rand(length) - 0.5) * 2
    vec_end = (rng_end.rand(length) - 0.5) * 2
    word2vec.add_vector('<PAD>', vec_pad)
#     word2vec.add_vector('<OOV>', vec_oov)
    word2vec.add_vector('<START>', vec_start)
    word2vec.add_vector('<END>', vec_end)
    return word2vec


def segmentation(word2vec, x):
    seg_list = jieba.cut(x)
#     token_id_list = [word2vec.key_to_index[x] if x in word2vec.key_to_index.keys() else word2vec.key_to_index['<OOV>'] for x in seg_list] 
    token_id_list = [word2vec.key_to_index[x] for x in seg_list if x in word2vec.key_to_index.keys() ] 
    return token_id_list


def create_dataset(contexts, 
                   titles, 
                   word2vec, 
                   max_len_encoder=config.MAX_LEN_ENCODER, 
                   max_len_decoder=config.MAX_LEN_DECODER):

    contexts = list(map(lambda x: segmentation(word2vec, x), contexts))

    encoder_input = tf.keras.preprocessing.sequence.pad_sequences(
        contexts, 
        value=word2vec.key_to_index['<PAD>'], 
        padding='post', 
        maxlen=max_len_encoder
    )
    encoder_input = encoder_input[:,::-1]                          # reverse, for better performance 
    
    if titles is None:
        return encoder_input
    else:    
        titles = list(map(lambda x: segmentation(word2vec, x), titles))
        decoder_input = [[word2vec.key_to_index['<START>']] + x for x in titles]
        decoder_output = [x + [word2vec.key_to_index['<END>']] for x in titles]
        decoder_input = tf.keras.preprocessing.sequence.pad_sequences(
            decoder_input, 
            value=word2vec.key_to_index['<PAD>'], 
            padding='post',
            maxlen=max_len_decoder
        )
        decoder_output = tf.keras.preprocessing.sequence.pad_sequences(
            decoder_output,
            value=word2vec.key_to_index['<PAD>'], 
            padding='post',
            maxlen=max_len_decoder
        )
        return encoder_input, decoder_input, decoder_output

    
                              
                
