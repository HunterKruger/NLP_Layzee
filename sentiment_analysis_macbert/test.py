import config
from dataset import CustomDataset
from model import create_model
from sklearn.metrics import classification_report
import os
import time
import tensorflow as tf
import pandas as pd
import joblib


    
def main():

    ## load label encoder
    print('Load label encoder...')
    le = joblib.load(config.LABEL_ENCODER_PATH)
    
    ## Model init
    mirrored_strategy = tf.distribute.MirroredStrategy()        # multi-GPU config
    with mirrored_strategy.scope():
        model_new = create_model(len(le.classes_))
        model_new.load_weights(config.CKPT_PATH)
    print('Initialization and loading finished.')
    
    ## Loading and processing data
    print('Loading and processing test set...')
    batch_size = config.BATCH_SIZE
    test_df = pd.read_csv(config.TEST_FILE)
    
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
    y_pred = model_new.predict(test_set, workers=config.CPUS)
    t2 = time.time()
    print('Inference finished.')
    print(str(round(t2 - t1,2)) + 's time consumption on ' + str(count_sample) + ' samples in ' + str(count_batch) + ' batches.') 
    print(str(round((t2 - t1)/count_sample, 4)) + 's per sample.')
    print(str(round((t2 - t1)/(count_batch), 4)) + 's per batch.')
    
    test_df['pred_proba'] = y_pred.max(axis=1)
    test_df['pred_class_encoded'] = y_pred.argmax(axis=1)
    test_df['pred_class']= le.inverse_transform(test_df['pred_class_encoded'])
    test_df['real_class']= le.inverse_transform(test_df[config.LABEL_FIELD])
    
    ## Evaluation
    print('Starting evaluation...')
    print(classification_report(test_df['real_class'], test_df['pred_class'], zero_division=0))
    test_df.to_csv(config.OUTPUT_TEST, index=False)
    print('Evaluation finished.')
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
