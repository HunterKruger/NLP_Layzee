import config
from dataset import CustomDataset, split_df
from model import create_model
import joblib
import os
import math
import datetime
from shutil import copyfile
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers.optimization_tf import create_optimizer
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import sys
import numpy as np
from sklearn.utils import class_weight
from focal_loss import SparseCategoricalFocalLoss 

# do not use all GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)                  

def main():
    
    ## Loading data
    print('Loading dataset...')
    df = pd.read_csv(config.INPUT_FILE, encoding='utf-8-sig')
    le = LabelEncoder()
    df[config.LABEL_FIELD] = le.fit_transform(df[config.LABEL_FIELD])
    joblib.dump(le, config.LABEL_ENCODER_PATH)
    
    train_df, val_df, test_df = split_df(
        df, 
        test_ratio=config.TEST_SIZE, 
        val_ratio=config.VALIDATION_SIZE, 
        target=None, 
        random_state=config.RANDOM_STATE
    )
    print('Training set shape: '+ str(train_df.shape))
    print('Validaiton set shape: '+ str(val_df.shape))
    print('Test set shape: ' + str(test_df.shape))
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
    print('Initializing model...')
    mirrored_strategy = tf.distribute.MirroredStrategy()     # multi-GPU config
    with mirrored_strategy.scope():
        model = create_model(nb_classes=len(le.classes_))
        print('Init a new model...')
        optimizer = Adam(config.LR)
        acc = SparseCategoricalAccuracy()
        loss = SparseCategoricalFocalLoss(gamma=config.GAMMA)
        model.compile(loss=loss, optimizer=optimizer, metrics=[acc])
    model.summary()
    print('Initialization finished.')
    

    ## Callbacks
    checkpoint_path = config.CKPT_PATH      
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_best_only=True, save_weights_only=True, verbose=1,  period=1)
    # stop training when there is no progress in 2 consecutive epochs
    early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=config.ES_PAT)
    # tensorboard
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=config.TENSORBOARD_DIR, update_freq='batch')    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=config.RD_LR_FAC, patience=config.RD_LR_PAT, min_lr=1e-6)
    
    print('Start training...')
    history = model.fit(
        train_set,
        validation_data=val_set,
        epochs=config.EPOCHS,
        workers=config.CPUS,
        max_queue_size=config.MAX_QUEUE_SIZE,
        callbacks=[checkpoint_callback, tb_callback, early_stop_callback, reduce_lr]
    )
    print('Training finished!')

    print('Saving config...')
    copyfile('config.py', config.CONFIG_FILE)         # save config
    print('Saving finished.')

if __name__ == "__main__":
    # execute only if run as a script
    main()