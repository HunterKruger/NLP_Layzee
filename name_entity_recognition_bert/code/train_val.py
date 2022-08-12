import os
import math
import numpy as np
import tensorflow as tf
import pandas as pd
import config
from dataset import CustomDataset
from model import create_model


os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES # specify GPU usage    

def main():
    
    ## Processing data
    print('Loading training and validation set...')
    train_set = CustomDataset(data=config.TRAINING_FILE, with_labels=True)
    val_set = CustomDataset(data=config.VALIDATION_FILE, with_labels=True)
    print('Loading finished.')
 
    ## Model init
    print('Initializing model...')
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = create_model(train_steps=math.ceil(train_set.get_sample_count()/config.EPOCHS) * config.EPOCHS)
    model.summary()
    print('Initialization finished.')

    ## Training and validation
    # save the model checkpoints
    checkpoint_path = config.CHECKPOINT_PATH
    # save weight but not model at the end of every 1 epoch
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, save_weights_only=True, verbose=1, period=1)
    # stop train when there is no progress in 1 consecutive epochs
    early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=1)
    # lr decay
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=1, min_lr=1e-5)
    # tensorboard
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=config.LOG_DIR)

    print('Start training...')
    history = model.fit(
        train_set,
        validation_data=val_set,                                                     
        epochs=config.EPOCHS,
        validation_batch_size=config.BATCH_SIZE,
        workers=config.CPUS,
        max_queue_size=config.MAX_QUEUE_SIZE,
        callbacks=[checkpoint_callback]
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

