import os
import time
import tensorflow as tf
import numpy as np
import config
from dataset import CustomDataset
from model import create_model
from evaluation import MltClsEvaluation
from seqeval.metrics import classification_report, accuracy_score, f1_score

os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES # specify GPU usage   


def prepare_for_token_eval(y_pred, tags_test, replacement='O'):
    # y_pred.shape=(sample_count,maxlen,label_count)
    # tags_test.shape=(sample_count,maxlen)
    
    # get y_pred and y_truth in 1-d array format
    y_pred_label = y_pred.argmax(axis=-1).flatten()
    y_truth = tf.squeeze(tags_test).numpy().flatten()    
    
    # remove CLS SEP PAD according to y_truth
    no_use_idx = np.where((y_truth == config.TAG2ID['PAD'])|(y_truth == config.TAG2ID['SEP'])|(y_truth == config.TAG2ID['CLS']))[0]
    y_truth_reduced = np.delete(y_truth, no_use_idx)
    y_pred_label_reduced = np.delete(y_pred_label, no_use_idx)

    # calibration: replace CLS SEP PAD by O in y_pred_label_reduced
    y_pred_label_reduced = np.where(y_pred_label_reduced==config.TAG2ID['PAD'], config.TAG2ID[replacement], y_pred_label_reduced) 
    y_pred_label_reduced = np.where(y_pred_label_reduced==config.TAG2ID['SEP'], config.TAG2ID[replacement], y_pred_label_reduced) 
    y_pred_label_reduced = np.where(y_pred_label_reduced==config.TAG2ID['CLS'], config.TAG2ID[replacement], y_pred_label_reduced) 
    return y_pred_label_reduced, y_truth_reduced


def prepare_for_tag_eval(y_pred, tags_test, replacement='O'):
    # illegal tags must be removed!
    # y_pred.shape=(sample_count,maxlen,label_count)
    # tags_test.shape=(sample_count,maxlen)
    
    y_pred_label = y_pred.argmax(axis=-1) 
    res_truth,res_pred=[],[]
    for i in range(len(tags_test)):
        re_pred, re_truth = [], []
        for j in range(len(tags_test[0])):
            if tags_test[i][j] not in [config.TAG2ID['PAD'],config.TAG2ID['CLS'],config.TAG2ID['SEP']]:
                re_truth.append(config.ID2TAG[tags_test[i][j]])
                # calibration: replace PAD CLS SEP by O in y_pred
                if y_pred_label[i][j] in [config.TAG2ID['PAD'],config.TAG2ID['CLS'],config.TAG2ID['SEP']]:
                    re_pred.append(replacement)
                else:
                    re_pred.append(config.ID2TAG[y_pred_label[i][j]])
        res_truth.append(re_truth)
        res_pred.append(re_pred)
            
    return res_pred,res_truth
    
    
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
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model_new = create_model(do_train=False)
        model_new.load_weights(config.ROOT_PATH + 'cp_000'+str(get_epoch)+'.ckpt').expect_partial()
    print('Initialization and loading finished.')

    ## Processing data
    print('Loading test set...')
    if mode == 1:
        test_set = CustomDataset(data=config.TEST_FILE, with_labels=True)
    if mode == 2:
        ## Read online data
        f = open(config.ONLINE_FILE)
        lines = f.read()
        online_data = lines.split('\n')
        f.close()
        ## temporarily write to file in conll format
        path = config.ONLINE_FILE_PRED
        with open(path, 'w') as f:
            for sent in online_data:
                for s in sent:
                    f.write(s+'\n')
                f.write('\n')
            f.close()
        test_set = CustomDataset(data=path, batch_size=config.ONLINE_BATCH_SIZE, with_labels=False)
    count_samples = test_set.get_sample_count()  # samples
    count_batches = len(test_set)   # batchs
    print('Loading finished.')

    
    ## Prediction
    print('Starting inference...')
    t1 = time.time()
    y_pred = model_new.predict(test_set)
    t2 = time.time()
    print('Inference finished.')
    print(str(round(t2-t1, 2)) + 's time consumption on ' + str(count_samples) + ' samples in ' + str(count_batches) + ' batch(es).')
    print(str(round((t2-t1)/count_samples,4)) + 's time consumption per sample.')
    print(str(round((t2-t1)/(count_batches),4)) + 's time consumption per batch.')

    if mode == 1:
        ## token-level evaluation
        print('Token-level evaluation:')
        t3 = time.time()
        tags_test = np.empty((0,config.MAX_LEN),int)
        for i, sample in enumerate(test_set):
            t = sample[1].numpy()
            tags_test = np.append(tags_test, t, axis=0)
        y_pred_tk, y_truth_tk = prepare_for_token_eval(y_pred[1], tags_test, 'O') 
        mle = MltClsEvaluation(y_pred_tk, y_truth_tk, config.UNIQUE_TAGS_LEGAL)
        mle.confusion_matrix(simple=True)
        t4 = time.time()
        print(str(round(t4-t3, 2)) + 's time consumption on whole data.')

        ## tag-level evaluation
        print('Tag-level evaluation:')
        t5 = time.time()
        res_pred, res_truth = prepare_for_tag_eval(y_pred[1], tags_test, 'O')
        result = classification_report(res_truth, res_pred)
        print(result)
        print("accuracy=" + str(round(accuracy_score(res_truth, res_pred), 2)))
        print("f1_score=" + str(round(f1_score(res_truth, res_pred), 2)))
        t6 = time.time()
        print(str(round(t6-t5, 2)) + 's time consumption on whole data.')
        
    if mode == 2:
        ## process prediction data
        y_pred_ = y_pred.argmax(axis=2) 
        res_pred=[]
        for i in range(len(y_pred)):
            re_pred = []
            length = tf.reduce_sum(test_set[0][1][i]).numpy()
            for j in range(1,length-2+1):
                if config.ID2TAG[y_pred_[i][j]] in ['SEP','CLS','PAD']:  # Calibration
                    re_pred.append('O')
                else:
                    re_pred.append(config.ID2TAG[y_pred_[i][j]])
            res_pred.append(re_pred)
        os.remove(path)  # remove temp file

        ## Write online data and prediction to conll file
        with open(path, 'w') as f:
            for i, sent in enumerate(online_data):
                for j, s in enumerate(sent):
                    f.write(s)
                    f.write(' ')
                    f.write(res_pred[i][j])
                    f.write('\n')
                f.write('\n')
            f.close()

        ## Write online data and prediction to trans file   
        if config.REQUIRE_TRANS:
            with open(config.ONLINE_FILE_PRED_TRANS, 'w') as f:
                for i, sentence in enumerate(online_data):
                    flag = False
                    tag = ''
                    entity = ''
                    for j, letter in enumerate(sentence):  
                        if res_pred[i][j]=='O':
                            if flag:
                                f.write('(' + entity + ':' + tag + ')')
                                flag = False
                            f.write(letter)
                        elif res_pred[i][j][0]=='B':
                            if flag:
                                f.write('(' + entity + ':' + tag + ')')
                            flag = True
                            tag = res_pred[i][j][2:]
                            entity = letter
                        else:         # 'I'
                            entity += letter
                    f.write('\n')

if __name__ == "__main__":
    # execute only if run as a script
    main()

