from config import config
from dataset import process_text, load_embedding, process_text_by_embedding
from evaluation import MltClsEvaluation

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
    if mode == 1:
        batch_size = config.BATCH_SIZE
        test_df = pd.read_csv(config.TEST_FILE)
    if mode == 2:
        batch_size = config.ONLINE_BATCH_SIZE
        f = open(config.ONLINE_FILE)
        lines = f.read()
        online_data = lines.split('\n')
        f.close()
        test_df = pd.DataFrame({'content':online_data})
        
    test_set = CustomDataset(
        test_df[config.CONTENT_FIELD].values.astype("str"), 
        test_df[config.LABEL_FIELD].values.astype("int"), 
    )
    count_sample = test_set.get_counts()
    count_batch = len(test_set)
    print('Loading finished.')

    
    ## Model init
    print('Initializing model and loading params...')
    if config.USE_RNN:
        model_new = tf.keras.models.load_model(config.MODEL_PATH)
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy()   # multi-GPU config
        with mirrored_strategy.scope():
            model_new = tf.keras.models.load_model(config.MODEL_PATH)    
    print('Initialization and loading finished.')
    
    ## Prediction
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # set tf logging detail display
    print('Starting inference...')
    t1 = time.time()
    y_pred = model_new.predict(test_set)
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
