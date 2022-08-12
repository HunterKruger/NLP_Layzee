import config
from dataset import CustomDataset
from model import create_model
from evaluation import MltClsEvaluation

import os
import time
import tensorflow as tf
import pandas as pd

# specify GPU usage   
os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES 

def main():
    
    mode = input('Choose a mode (1 for evaluation, 2 for only inference):')
    mode = int(mode)
    
    ## Model init
    print('Initializing model and loading params...')
    filenames = dict()
    i = 0
    for f in os.listdir(config.ROOT_PATH):
        if '.hdf5' in f:
            filenames[i+1] = f
            i+=1
    print(filenames)
    get_epoch = input('Choose a checkpoint:')
    get_epoch = int(get_epoch)
    mirrored_strategy = tf.distribute.MirroredStrategy()   # multi-GPU config
    with mirrored_strategy.scope():
        model_new = create_model()
        model_new.load_weights(config.ROOT_PATH + filenames[get_epoch])
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
