from config import config
from dataset import CustomDataset
from model import create_model
from evaluation import BinClsEvaluation

import os
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES    # specify GPU usage    

def main():
    
    ##  Environment check and config
    print('Environment check and config...')
    warnings.filterwarnings("ignore")
    print('TensorFlow Version:')
    print(tf.__version__)
    print('GPU status:')
    print(tf.test.is_gpu_available())
    print('Environment check and config finished.')
    
    mode = input('Choose a mode (1 for test, 2 for online inference):')
    mode = int(mode)
    
    ## Model init
    print('Initializing model and loading params...')
    filenames = []
    for f in os.listdir(config.ROOT_PATH):
        if '.index' in f:
            filenames.append(f)
    print(filenames)
    get_epoch = input('Choose an epoch:')
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model_new  = create_model(do_train=False, summary=False)
        model_new.load_weights(config.ROOT_PATH + 'cp_000'+str(get_epoch)+'.ckpt').expect_partial()
    print('Initialization and loading finished.')
    
    ## Loading and processing data
    print('Loading and processing test set...')
    if mode == 1:
        test_df = pd.read_csv(config.TEST_FILE)
        test_set = CustomDataset(
            sentence_pairs=test_df[[config.SENTENCE_FIELD,config.SENTENCE_FIELD2]].values.astype("str"),
            labels=test_df[config.LABEL_FIELD],
            batch_size=config.BATCH_SIZE
        )
    if mode == 2:
        ## Read online data
        f = open(config.ONLINE_TEST_FILE)
        lines = f.read()
        online_data = lines.split('\n')
        online_data_s1, online_data_s2 = [], []
        for pair in online_data:
            s1s2 = pair.split('<>')
            online_data_s1.append(s1s2[0])
            online_data_s2.append(s1s2[1])
        f.close()
        test_df = pd.DataFrame({'s1':online_data_s1, 's2':online_data_s2})
        test_set = CustomDataset(
            sentence_pairs=test_df[[config.SENTENCE_FIELD, config.SENTENCE_FIELD2]].values.astype("str"),
            batch_size=config.ONLINE_BATCH_SIZE
        )
    count_sample = test_set.get_counts()
    count_batch = len(test_set)
    print('Loading finished.')

    ## Prediction
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)     # set tf logging detail display
    print('Starting inference...')
    t1 = time.time()
    y_pred = model_new.predict(test_set)
    t2 = time.time()
    print('Inference finished.')
    print(str(round(t2-t1, 2)) + 's time consumption on ' + str(count_sample) + ' samples in ' + str(count_batch) + ' batch(es).')
    print(str(round((t2 - t1)/count_sample, 4)) + 's per sample.')
    print(str(round((t2 - t1)/(count_batch), 4)) + 's per batch.')
    test_df['pred_proba']=y_pred

    if mode == 1:
        ## Evaluation
        print('Starting evaluation...')
        t3 = time.time()
        bce = BinClsEvaluation(test_df.pred_proba, test_df.label)
        bce.confusion_matrix()
        bce.detailed_metrics()
        t4 = time.time()
        test_df['pred_label'] = [1 if x>=bce.best_cutoff else 0 for x in test_df['pred_proba']]
        test_df.to_csv(config.OUTPUT_TEST, index=False)
        print('Evaluation finished, ' + str(round(t4 - t3,4)) + 's time consumption.')
    if mode == 2:
        test_df.to_csv(config.OUTPUT_ONLINE, index=False)
        print('File saved to '+config.OUTPUT_ONLINE+'.')


if __name__ == "__main__":
    # execute only if run as a script
    main()