import config
from dataset import CustomDataset, train_test_split, undersampling, load_embedding
from model import create_siamese_model

import os
import time
import tensorflow as tf
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
        data_df = pd.read_csv(config.ALL_FILE)
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
    if config.UNDER_SAMPLING:
        train_df = undersampling(train_df)
        val_df = undersampling(val_df)
    
    ## Processing data
    print('Processing dataset...')
    t1 = time.time()
    word2vec = load_embedding(config.WORD2VEC_FILE)

    train_set = CustomDataset(
        train_df[config.SENTENCE_FIELD].values.astype('str'), 
        train_df[config.SENTENCE_FIELD2].values.astype('str'), 
        train_df[config.LABEL_FIELD].values.astype('float32'), 
        word2vec
    )
    val_set = CustomDataset(
        val_df[config.SENTENCE_FIELD].values.astype('str'), 
        val_df[config.SENTENCE_FIELD2].values.astype('str'), 
        val_df[config.LABEL_FIELD].values.astype('float32'), 
        word2vec
    )
    t2 = time.time()
    print('Processing finished, time consumption in ' + str(t2-t1) + 's.')

    ## Model init
    print('Initializing model...')
    if config.USE_RNN:
        model = create_siamese_model(embedding_matrix=word2vec, use_rnn=True, fake_siamese=config.FAKE_SIAMESE, do_train=True)
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy()   # multi-GPU config
        with mirrored_strategy.scope():
            model = create_siamese_model(embedding_matrix=word2vec, use_rnn=False, fake_siamese=config.FAKE_SIAMESE, do_train=True)
    print('Initialization finished.')

    
    ## Training and validation
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=config.MODEL_PATH, 
        monitor='val_auc', 
        mode='max', 
        save_best_only=True, 
        save_weights_only=False)
    plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        mode='min', 
        factor=config.LR_REDUCE_FACTOR, 
        patience=config.LR_REDUCE_PATIENCE)
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        patience=config.EARLY_STOP_PATIENCE, 
        monitor='val_auc',
        mode='max')
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=config.LOG_DIR)     

    print('Start training...')
    history = model.fit(
        train_set,
        validation_data=val_set,
        epochs=config.EPOCHS,
        workers=config.CPUS,
        max_queue_size=config.MAX_QUEUE_SIZE,
        callbacks=[checkpoint_callback, tb_callback, plateau_callback, early_stop_callback]
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


