import config
from dataset import process_text
from model import create_model

import os
import time
import tensorflow as tf
import numpy as np
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES    # specify GPU usage    

def main():
    

    ## Processing data
    print('Loading and processing dataset...')
    t1 = time.time()
    X_train, y_train, tokenizer = process_text(
        config.TRAINING_FILE, 
        maxlen=config.MAX_LEN, 
        tokenizer_path=config.TOKENIZER_PATH,
        fit_transform=True,
        with_labels=True 
    )
    X_val, y_val, tokenizer = process_text(
        config.VALIDATION_FILE, 
        maxlen=config.MAX_LEN, 
        tokenizer_path=config.TOKENIZER_PATH,
        fit_transform=False,
        with_labels=True
    )    
    t2 = time.time()
    print('Loading and processing finished, time consumption in ' + str(t2-t1) + 's.')

    ## Model init
    print('Initializing model...')
    model = create_model(vocab_size=len(tokenizer.word_index)+1)   # +1 for PAD
    model.summary()
    print('Initialization finished.')
    
    ## Training and validation
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=config.MODEL_PATH, 
        monitor='val_decode_sequence_acc', 
        mode='max', 
        save_best_only=True, 
        save_weights_only=False)
    plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=config.LR_REDUCE_FACTOR, 
        patience=config.LR_REDUCE_PATIENCE)
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        patience=config.EARLY_STOP_PATIENCE, 
        monitor='val_decode_sequence_acc',
        mode='max')
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=config.LOG_DIR, update_freq='batch') 
    
    print('Start training...')
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val,y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_batch_size=config.BATCH_SIZE,
        callbacks=[checkpoint_callback, plateau_callback, tb_callback, early_stop_callback]
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



