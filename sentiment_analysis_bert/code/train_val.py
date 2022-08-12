import config
from dataset import CustomDataset, process_data, train_test_split
from model import create_model

import os
import math
import datetime
from shutil import copyfile
import tensorflow as tf
import pandas as pd
from transformers.optimization_tf import create_optimizer
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# specify GPU ids  
os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES   

# do not use all GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)                  


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
    filenames = dict()
    i = 1
    for f in os.listdir(config.ROOT_PATH):
        if '.hdf5' in f:
            filenames[i] = f
            i+=1
    print(filenames)
    get_epoch = input('Choose a checkpoint (input 0 to train a new model):')
    get_epoch = int(get_epoch)

    print('Initializing model...')
    mirrored_strategy = tf.distribute.MirroredStrategy()     # multi-GPU config
    with mirrored_strategy.scope():
        model = create_model()
        if get_epoch == 0:     
            print('Init a new model...')
            checkpoint_path = config.ROOT_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-{epoch:02d}.hdf5'                 # save the model checkpoints
            optimizer, lr_schedule = create_optimizer(
                init_lr=config.ADAM_LR, 
                num_train_steps=math.ceil(train_df.shape[0]/config.EPOCHS)*config.EPOCHS,
                num_warmup_steps=config.ADAM_WARMUP_STEPS,
                min_lr_ratio=config.ADAM_MIN_LR_RATIO,
                weight_decay_rate=config.ADAM_DECAY_RATE,
                power=config.ADAM_POWER
            )
        else:
            print('Init model from a checkpoint...')
            checkpoint_path = config.ROOT_PATH + filenames[get_epoch].replace('.hdf5','') + '__' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-{epoch:02d}.hdf5'   # save the model checkpoints
            model.load_weights(config.ROOT_PATH + filenames[get_epoch])
            lr_schedule = ExponentialDecay(
                config.SGD_LR,
                decay_steps=config.SGD_DECAY_STEPS,
                decay_rate=config.SGD_DECAY_RATE
            )
            optimizer = SGD(
                learning_rate=lr_schedule, 
                momentum=config.SGD_MOMENTUM, 
                nesterov=config.SGD_NESTEROV
            )           
        metric = SparseCategoricalAccuracy()
        loss = SparseCategoricalCrossentropy()
        model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    model.summary()
    print('Initialization finished.')

    ## Callbacks
    # save weight but not model at the end of every 1 epoch
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1,  period=1)
    # stop training when there is no progress in 1 consecutive epochs
    early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=1)
    # tensorboard
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=config.TENSORBOARD_DIR, update_freq='batch')     

    print('Start training...')
    history = model.fit(
        train_set,
        validation_data=val_set,
        epochs=config.EPOCHS,
        workers=config.CPUS,
        max_queue_size=config.MAX_QUEUE_SIZE,
        callbacks=[checkpoint_callback, tb_callback, early_stop_callback]
    )
    print('Training finished!')

    print('Saving training history and config...')
    hist_df = pd.DataFrame(history.history) 
    hist_df.to_csv(config.HISTORY_FILE, index=False)  # save training history
    copyfile('config.py', config.CONFIG_FILE)         # save config
    print('Saving finished.')

if __name__ == "__main__":
    # execute only if run as a script
    main()