from config import config
from dataset import CustomDataset
from model import create_model
from evaluation import MltClsEvaluation

import os
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES # specify GPU usage   

def main():
    
    
    mode = input('Choose a mode (1 for test, 2 for online inference):')
    mode = int(mode)
    
    ## Model init
    print('Initializing model and loading params...')
    phase = input('Choose a phase (1 or 2): ')
    phase = int(phase)
    filenames = []
    for f in os.listdir(config.ROOT_PATH if phase==1 else config.ROOT_PATH2):
        if '.index' in f:
            filenames.append(f)
    print(filenames)
    get_epoch = input('Choose an epoch:')
    if phase == 1:
        model_new, mirrored_strategy = create_model(do_train=False, summary=False, phase=1)
        with mirrored_strategy.scope():
            model_new.load_weights(config.ROOT_PATH + 'cp_000'+str(get_epoch)+'.ckpt').expect_partial()
    if phase == 2: 
        model_new, mirrored_strategy = create_model(do_train=False, summary=False, phase=2)
        with mirrored_strategy.scope():
            model_new.load_weights(config.ROOT_PATH2 + 'cp_000'+str(get_epoch)+'.ckpt').expect_partial()
    print('Initialization and loading finished.')
    
    ## Loading and processing data
    print('Loading and processing test set...')
    if mode == 1:
        batch_size = config.BATCH_SIZE
        test_df = pd.read_csv(config.TEST_FILE)
    if mode == 2:
        f = open(config.ONLINE_FILE)
        lines = f.read()
        online_data = lines.split('\n')
        f.close()
        test_df = pd.DataFrame({'content':online_data})
        batch_size = config.ONLINE_BATCH_SIZE
    test_set = CustomDataset(
        sentences=test_df[config.CONTENT_FIELD].values.astype("str"),
        labels=None,
        batch_size=batch_size,
        shuffle=False
    )
    count_sample = test_set.get_size()
    count_batch = len(test_set) 
    print('Loading finished.')

    ## Prediction
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # set tf logging detail display
    print('Starting inference...')
    t1 = time.time()
    y_pred = model_new.predict(test_set)
    y_pred_label = y_pred.argmax(axis=1)
    t2 = time.time()
    print('Inference finished.')
    print(str(round(t2 - t1,2)) + 's time consumption on ' + str(count_sample) + ' samples in ' + str(count_batch) + ' batches.') 
    print(str(round((t2 - t1)/count_sample,4)) + 's per sample.')
    print(str(round((t2 - t1)/(count_batch),4)) + 's per batch.')
    
    test_df['pred']=y_pred.argmax(axis=1)
    test_df['pred_cls']=test_df['pred'].map(config.IDX2CLS)
    
    if mode == 2:
        test_df.to_csv(config.OUTPUT_ONLINE, index=False)
        print('File saved to '+config.OUTPUT_ONLINE+'.')
    
    ## Evaluation
    if mode == 1:
        print('Starting evaluation...')
        mle = MltClsEvaluation(test_df.pred, test_df.class_label, config.LABELS)
        mle.confusion_matrix()
        mle.detailed_metrics()
        test_df.to_csv(config.OUTPUT_TEST, index=False)
        print('Evaluation finished.')
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
