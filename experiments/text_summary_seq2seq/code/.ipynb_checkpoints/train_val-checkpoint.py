import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from config import config
from dataset import create_dataset, load_embedding
from model import seq2seq, rebuild_encoder_decoder

# from model3 import Seq2Seq, Encoder, Decoder, encoder_infer, decoder_infer

os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES    # specify GPU usage   


def main():
    
    ## Loading data
    if config.ALREADY_SPLIT: 
        print('Loading dataset...')
        train_df = pd.read_csv(config.TRAIN_FILE) 
        val_df = pd.read_csv(config.VALIDATION_FILE) 
        print('Training set shape: '+ str(train_df.shape))
        print('Validation set shape: '+ str(val_df.shape))
        print('Loading finished.')
    else:
        data_df =  pd.read_csv(config.ALL_FILE)
        train_df = data_df[:config.TRAIN_SIZE]
        val_df = data_df[config.TRAIN_SIZE:config.TRAIN_SIZE+config.VALIDATION_SIZE]
        test_df = data_df[config.TRAIN_SIZE+config.VALIDATION_SIZE:config.TRAIN_SIZE+config.VALIDATION_SIZE+config.TEST_SIZE]
        print('Training set shape: '+ str(train_df.shape))
        print('Validation set shape: '+ str(val_df.shape))
        print('Test set shape: '+ str(test_df.shape))
        print('Loading finished.')
        print('Saving training set & test set to local...')
        train_df.to_csv(config.TRAIN_FILE, index=False)
        val_df.to_csv(config.VALIDATION_FILE, index=False)
        test_df.to_csv(config.TEST_FILE, index=False)
        print('Saving finished.')
        
        
    ## Processing data
    print('Processing dataset...')
    word2vec = load_embedding()
    train_encoder_input, train_decoder_input, train_decoder_output = create_dataset(
        contexts=train_df[config.CONTENT_FIELD].values.astype('str'), 
        titles=train_df[config.TITLE_FIELD].values.astype('str'),
        word2vec=word2vec
    )
    val_encoder_input, val_decoder_input, val_decoder_output  = create_dataset(
        contexts=val_df[config.CONTENT_FIELD].values.astype('str'), 
        titles=val_df[config.TITLE_FIELD].values.astype('str'),
        word2vec=word2vec
    )
    print('Processing finished.')
 
    ## Modeling
    print('Initializing model...')    
    model = seq2seq(word2vec)
    model.summary()
    print('Initialization finished.')

    ## Training and validation
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=config.MODEL_PATH, 
        monitor='val_sparse_categorical_accuracy',
        mode='max', 
        save_best_only=True, 
        save_weights_only=False)
    plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=config.LR_REDUCE_FACTOR, 
        patience=config.LR_REDUCE_PATIENCE)
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        patience=config.EARLY_STOP_PATIENCE,
        monitor='val_sparse_categorical_accuracy', 
        mode='max')
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=config.LOG_DIR)      

    print('Start training...')
    history = model.fit(
        x=[train_encoder_input, train_decoder_input],
        y=train_decoder_output,
        validation_data=([val_encoder_input, val_decoder_input],val_decoder_output),
        epochs=config.EPOCHS,
        callbacks=[checkpoint_callback, tb_callback, plateau_callback, early_stop_callback]
    )
    print('Training finished!')

    print('Saving training history to csv...')
    hist_df = pd.DataFrame(history.history) 
    with open(config.HISTORY_FILE, mode='w') as f:
        hist_df.to_csv(f, index=False)
    print('Saving finished.')
    
    model = tf.keras.models.load_model(config.MODEL_PATH)
    model.summary()
    ## rebuild encoder and decoder
    print('Rebuild encoder and decoder...')
    encoder_model, decoder_model = rebuild_encoder_decoder(model)
    encoder_model.summary()
    decoder_model.summary()
    encoder_model.save(config.ENCODER_PATH)
    decoder_model.save(config.DECODER_PATH)
    print('Encoder and decoder rebuilt and saved...')

    
if __name__ == "__main__":
    # execute only if run as a script
    main()


