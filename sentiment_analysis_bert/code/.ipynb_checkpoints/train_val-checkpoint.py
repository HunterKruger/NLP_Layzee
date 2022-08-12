from config import config
from dataset import CustomDataset, process_data, train_test_split
from model import create_model

import os
import math
import tensorflow as tf
import warnings
import numpy as np
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES    # specify GPU usage    

def main():
    

    ## Loading data
    print('Loading dataset...')
    if config.ALREADY_SPLIT:
        train_df = pd.read_csv(config.TRAIN_FILE) 
        val_df = pd.read_csv(config.VALIDATION_FILE)    
        print('Training set shape: '+ str(train_df.shape))
        print('Validaiton set shape: '+ str(val_df.shape))
        print('Loading finished.')
    else:
        data_df = process_data(config.INPUT_FILE, config.CLS2IDX, True)     # DataFrame, only used labeled data
        train_df, test_df = train_test_split(
            data_df, 
            test_size=config.TEST_SIZE, 
            shuffle=True, 
            random_state=config.RANDOM_STATE)
        train_df, val_df = train_test_split(
            train_df, 
            test_size=config.VALIDATION_SIZE, 
            shuffle=True, 
            random_state=config.RANDOM_STATE)  
        print('Training set shape: '+ str(train_df.shape))
        print('Validaiton set shape: '+ str(val_df.shape))
        print('Test set shape: '+ str(test_df.shape))
        print('Loading finished.')
        print('Saving training set & validation set & test set to local...')
        train_df.to_csv(config.TRAIN_FILE, index=False)
        val_df.to_csv(config.VALIDATION_FILE, index=False)
        test_df.to_csv(config.TEST_FILE, index=False)
        print('Saving finished.')
    

    ## Processing data
    print('Processing dataset...')
    train_set = CustomDataset(
        sentences=train_df[config.CONTENT_FIELD].values.astype("str"),
        labels=train_df[config.LABEL_FIELD],
        batch_size=config.BATCH_SIZE
    )
    val_set = CustomDataset(
        sentences=val_df[config.CONTENT_FIELD].values.astype("str"),
        labels=val_df[config.LABEL_FIELD],
        batch_size=config.BATCH_SIZE
    )
    print('Processing finished.')

 
    ## Model init
    phase = input('Choose a phase (1 or 2): ')
    phase = int(phase)
    print('Initializing model...')
    if phase == 1:
        checkpoint_path = config.CHECKPOINT_PATH       # save the model checkpoints
        mirrored_strategy = tf.distribute.MirroredStrategy()   # multi-GPU config
        with mirrored_strategy.scope():
            model = create_model(do_train=True, phase=1, train_steps=math.ceil(train_df.shape[0]/config.EPOCHS)*config.EPOCHS)
    if phase == 2:
        checkpoint_path = config.CHECKPOINT_PATH2      # save the model checkpoints
        filenames = []
        for f in os.listdir(config.ROOT_PATH):
            if '.index' in f:
                filenames.append(f)
        print(filenames)
        get_epoch = input('Choose an epoch in phase 1:')
        mirrored_strategy = tf.distribute.MirroredStrategy()   # multi-GPU config
        with mirrored_strategy.scope():
            model = create_model(do_train=True, phase=2)
            model.load_weights(config.ROOT_PATH + 'cp_000'+str(get_epoch)+'.ckpt').expect_partial()
    model.summary()
    print('Initialization finished.')

    ## Training and validation
    # save weight but not model at the end of every 1 epoch
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1,  period=1)
    # stop training when there is no progress in 1 consecutive epochs
    early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=1)
    # tensorboard
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=config.LOG_DIR)     

    print('Start training...')
    history = model.fit(train_set,
                        validation_data=val_set,
                        epochs=config.EPOCHS,
                        batch_size=config.BATCH_SIZE,
                        validation_batch_size=config.BATCH_SIZE,
                        workers=config.CPUS,
                        max_queue_size=config.MAX_QUEUE_SIZE,
                        callbacks=[checkpoint_callback, tb_callback]
                       )
    print('Training finished!')
    
    print('Saving training history to csv...')
    hist_df = pd.DataFrame(history.history) 
    with open(config.HISTORY_FILE, mode='w') as f:
        hist_df.to_csv(f, index=False)
    print('Saving finished.')


if __name__ == "__main__":
    # execute only if run as a script
    main()


