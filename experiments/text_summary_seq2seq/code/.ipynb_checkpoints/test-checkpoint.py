import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge 

from config import config
from dataset import create_dataset, load_embedding
from model import seq2seq

os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES    # specify GPU usage   


def greedy_prediction(test_encoder_input, test_decoder_output, encoder, decoder, word2vec):
    
    states_value = encoder.predict(test_encoder_input)     # [state h, state_c]
    target_seq = np.zeros((config.TEST_SIZE, config.MAX_LEN_DECODER))
    target_seq[:,0] = word2vec.key_to_index['<START>']
    decoded_sentence = ""
    decoded_sentences = [decoded_sentence]*config.TEST_SIZE
    for i in range(config.MAX_LEN_DECODER):
        output_tokens, h, c = decoder.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens, axis=-1)[:,0]
        sampled_char = [word2vec.index_to_key[x] for x in sampled_token_index]
        
        decoded_sentences_new=[]
        for i, decoded_sent in enumerate(decoded_sentences):
            decoded_sent += sampled_char[i]
            decoded_sentences_new.append(decoded_sent+' ')
        decoded_sentences = decoded_sentences_new

        target_seq = np.zeros((config.TEST_SIZE,config.MAX_LEN_DECODER))
        target_seq[:,0] = word2vec.key_to_index['<START>']

        states_value = [h, c]

    decoded_sentences_cleaned = []
    for i, sample in enumerate(decoded_sentences):
        position = sample.find('<EOS>')
        decoded_sentences_cleaned.append(sample[:position])
        
        
    y_pred_list = [x.split(' ') for x in decoded_sentences_cleaned]   # list of list
    y_true_list = []                                                  # list of list
    for i, sample in enumerate(test_decoder_output):
        sent = [word2vec.index_to_key[x] for x in sample if word2vec.index_to_key[x] not in ['<PAD>','<END>']]
        y_true_list.append(sent)

    y_pred_str = decoded_sentences_cleaned                           # list of str with ' ' splitted
    y_true_str = [' '.join(x) for x in y_true_list]                  # list of str with ' ' splitted
 
            
    return y_pred_list, y_true_list, y_pred_str, y_true_str
    

def main():
    
    ## Loading data
    print('Loading and processing test set...')
    word2vec = load_embedding()
    test_df = pd.read_csv(config.TEST_FILE) 
    test_encoder_input, test_decoder_input, test_decoder_output = create_dataset(
        contexts=test_df[config.CONTENT_FIELD].values.astype('str'), 
        titles=test_df[config.TITLE_FIELD].values.astype('str'),
        word2vec=word2vec
    )
    print('Loading and processing finished.')
 

    ## Model init
    print('Initializing model and loading params...')
    encoder = tf.keras.models.load_model(config.ENCODER_PATH)
    encoder.summary()
    decoder = tf.keras.models.load_model(config.DECODER_PATH)
    decoder.summary()
    print('Initialization and loading finished.')

    ## Prediction
    y_pred_list, y_true_list, y_pred_str, y_true_str = greedy_prediction(test_encoder_input, test_decoder_output, encoder, decoder, word2vec)
        
    
    ## Evaluation
    bleu = corpus_bleu(y_pred_list, y_true_list)
    print('Bleu score: ' + str(bleu))
    
    print('Rouge scores: ')
    rouge = Rouge()
    scores_rouge = rouge.get_scores(y_pred_str, y_true_str, avg=True)
    print(scores_rouge)

    
if __name__ == "__main__":
    # execute only if run as a script
    main()


