import config
from dataset import CustomDataset,load_embedding
from evaluation import BinClsEvaluation, BinClsEvaluation
from model import loss, euclidean_distance

import os
import time
import tensorflow as tf
import numpy as np
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES # specify GPU usage   

def main():

    ## Loading and processing data
    print('Loading and processing test set...')
    mode = input('Choose a mode (1 for test, 2 for online inference):')
    mode = int(mode)
    word2vec = load_embedding(config.WORD2VEC_FILE)

    if mode == 1:
        test_df = pd.read_csv(config.TEST_FILE)
        test_set = CustomDataset(
            test_df[config.SENTENCE_FIELD].values.astype('str'), 
            test_df[config.SENTENCE_FIELD2].values.astype('str'), 
            test_df[config.LABEL_FIELD].values.astype('int'), 
            word2vec,
            batch_size=config.BATCH_SIZE
        )
    if mode == 2:
        f = open(config.ONLINE_FILE)
        lines = f.read()
        online_data = lines.split('\n')
        online_data_s1, online_data_s2 = [], []
        for pair in online_data:
            s1s2 = pair.split
            online_data_s1.append(s1s2[0])
            online_data_s2.append(s1s2[1])
        f.close()
        test_df = pd.DataFrame({config.SENTENCE_FIELD:online_data_s1, config.SENTENCE_FIELD2:online_data_s2})
        test_set = CustomDataset(
            test_df[config.SENTENCE_FIELD].values.astype('str'), 
            test_df[config.SENTENCE_FIELD2].values.astype('str'), 
            None, 
            word2vec,
            batch_size=config.ONLINE_BATCH_SIZE
        )

    count_sample = test_set.get_counts()
    count_batch = len(test_set)
    print('Loading finished.')

    
    ## Model init
    print('Initializing model and loading params...')
    if config.USE_RNN:
        model_new = tf.keras.models.load_model(config.MODEL_PATH, custom_objects={'contrastive_loss':loss, 'euclidean_distance':euclidean_distance})
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy()   # multi-GPU config
        with mirrored_strategy.scope():
            model_new = tf.keras.models.load_model(config.MODEL_PATH, custom_objects={'contrastive_loss':loss, 'euclidean_distance':euclidean_distance})
    print('Initialization and loading finished.')
    
    ## Prediction
    print('Starting inference...')
    t1 = time.time()
    y_pred = model_new.predict(test_set)
    t2 = time.time()
    print('Inference finished.')
    print(str(round(t2 - t1,2)) + 's time consumption on ' + str(count_sample) + ' samples in ' + str(count_batch) + ' batches.') 
    print(str(round((t2 - t1)/count_sample,4)) + 's per sample.')
    print(str(round((t2 - t1)/(count_batch),4)) + 's per batch.')
    
    test_df['pred_proba']=y_pred
    
    if mode == 2:
        test_df.to_csv(config.OUTPUT_ONLINE, index=False)
        print('File saved to '+config.OUTPUT_ONLINE+'.')
    
    ## Evaluation
    if mode == 1:
        print('Starting evaluation...')
        eva = BinClsEvaluation(test_df['pred_proba'], test_df[config.LABEL_FIELD])
        eva.confusion_matrix()
        eva.detailed_metrics()
        test_df.to_csv(config.OUTPUT_TEST, index=False)
        print('Evaluation finished.')
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
