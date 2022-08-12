import config
from dataset import CustomDataset
from model import create_model

import os
import time
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES # specify GPU usage   

def main():
    
    ## Model init
    print('Initializing model and loading params...')
    
    filenames = dict()
    i = 0
    for f in os.listdir(config.ROOT_PATH):
        if '.index' in f:
            filenames[i+1] = f
            i += 1
    print(filenames)
    get_epoch = input('Choose a checkpoint:')
    get_epoch = int(get_epoch)

    mirrored_strategy = tf.distribute.MirroredStrategy()   # multi-GPU config
    with mirrored_strategy.scope():
        model_new = create_model(do_train=False, phase=1)
        model_new.load_weights(config.ROOT_PATH + filenames[get_epoch].replace('.index','')).expect_partial() 
    print('Initialization and loading finished.')
    
    while True:
        sample = input('Please input a sentence (e to Exit):')
        if sample == 'e':
            break
        t1 = time.time()
        encoded = CustomDataset([sample],shuffle=False,batch_size=1)
        ## Prediction
        y_pred = model_new.predict(encoded)
        y_pred_label = y_pred[0].argmax(axis=-1)
        t2 = time.time()
        print(config.IDX2CLS[y_pred_label])
        print(str(round(t2 - t1,2)) + 's time consumption')

    
if __name__ == "__main__":
    # execute only if run as a script
    main()
