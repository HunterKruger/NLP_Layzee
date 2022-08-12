from config import config
from dataset import CustomDataset
from model import create_model

import os
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES    # specify GPU usage    


def exact_match(test_df):
    pred = (test_df.answer == test_df.pred_answer).to_numpy()
    result = pred.sum()/len(test_df)
    print('Exact match: ' + str(result))

    
def get_f1(test_df):
    sum_intersection = 0
    sum_pred = 0
    sum_real = 0
    for i in range(len(test_df)):
        pred_ans = test_df.pred_answer[i]
        real_ans = test_df.answer[i]
        len_pred_ans = len(pred_ans)
        len_real_ans = len(real_ans)
        len_intersection = len(set(pred_ans).intersection(real_ans))
        sum_intersection += len_intersection
        sum_pred += len_pred_ans
        sum_real += len_real_ans
    precision = sum_intersection/sum_pred
    recall = sum_intersection/sum_real
    f1 = 2*precision*recall/(precision+recall)
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1: ' + str(f1))
    
    
def get_pred_ans(start,end,document):
    return document[start:end]
    
def main():

    
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
    mirrored_strategy = tf.distribute.MirroredStrategy()   # multi-GPU config
    with mirrored_strategy.scope():
        model_new = create_model(do_train=False, summary=False)
        model_new.load_weights(config.ROOT_PATH + 'cp_000'+str(get_epoch)+'.ckpt').expect_partial()
    print('Initialization and loading finished.')
    
    ## Loading and processing data
    print('Loading and processing test set...')
    if mode == 1:
        test_df = pd.read_csv(config.TEST_FILE)
        test_set = CustomDataset(
            questions=test_df[config.QUESTION_FIELD].values.astype("str"),
            documents=test_df[config.DOCUMENT_FIELD].values.astype("str"),
            starts=test_df[config.START_FIELD].values.astype("int"),
            ends=test_df[config.END_FIELD].values.astype("int")
        )
    if mode == 2:
        ## Read online data
        f = open(config.ONLINE_FILE)
        lines = f.read()
        online_data = lines.split('\n')
        documents, questions = [], []
        for pair in online_data:
            s1s2 = pair.split('    ')
            documents.append(s1s2[0])
            questions.append(s1s2[1])
        f.close()
        test_df = pd.DataFrame({'document':documents, 'question':questions})
        test_set = CustomDataset(
            questions=test_df[config.QUESTION_FIELD].values.astype("str"),
            documents=test_df[config.DOCUMENT_FIELD].values.astype("str"),
            with_labels=False
        )
    count_sample = test_set.get_sample_count()
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

    # Processing prediction
    pred_starts, pred_ends = y_pred[0].argmax(axis=1)-1, y_pred[1].argmax(axis=1)-1
    test_df['pred_start'] = pred_starts
    test_df['pred_end'] = pred_ends
    test_df['pred_answer'] = test_df.apply(lambda x: get_pred_ans(x['pred_start'] ,x['pred_end'], x['document']), axis=1)
    
    if mode == 1:
        test_df.to_csv(config.OUTPUT_TEST, index=False)
        ## Evaluation
        print('Starting evaluation...')
        t3 = time.time()
        exact_match(test_df)
        get_f1(test_df)
        t4 = time.time()
        print('Evaluation finished, ' + str(round(t4-t3,4)) + 's time consumption.')
    if mode == 2:
        test_df.to_csv(config.OUTPUT_ONLINE, index=False)
        print('File saved to '+config.OUTPUT_ONLINE+'.')


if __name__ == "__main__":
    # execute only if run as a script
    main()