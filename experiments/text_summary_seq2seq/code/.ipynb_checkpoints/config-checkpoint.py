'''
Text summarization in Chinese.

CPU
Use pretrained embedding
Seq2Seq Model
Use only the last hidden state of encoder
No attention
Reverse Input
Teacher Forcing
Greedy search for inference
Bleu & Rouge Score for evalution

Poor performance in evaluation due to the input & output length.
'''



import datetime

class config:
    
    # GPU
    CUDA_VISIBLE_DEVICES = '-1'

    # Model Hyperparameters
    MAX_LEN_ENCODER = 100
    MAX_LEN_DECODER = 30
    HIDDEN_STATES_ENCODER = 250
    HIDDEN_STATES_DECODER = 250
    BATCH_SIZE = 32
    EPOCHS = 16
    LEARNING_RATE = 0.01
    EARLY_STOP_PATIENCE = 2
    LR_REDUCE_FACTOR = 0.5
    LR_REDUCE_PATIENCE = 1
    RECURRENT_DROPOUT = 0.2
    CLIPNORM = 1

    # Model outputs
    ROOT_PATH = '../model/'
    MODEL_PATH = ROOT_PATH+'seq2seq.h5'
    ENCODER_PATH = ROOT_PATH + 'encoder.h5'
    DECODER_PATH = ROOT_PATH + 'decoder.h5'

    # Log outputs
    LOG_DIR = "../log/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")                 # log for tensorboard
    HISTORY_FILE = '../log/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv'     # log for training history

    # Data inputs
    ALREADY_SPLIT = False
    TRAIN_SIZE = 20000
    VALIDATION_SIZE = 2000
    TEST_SIZE = 500
    ORIGIN_FILE = '../origin_data/lcsts_data.json'
    ALL_FILE = '../data/all.csv'
    TRAIN_FILE = '../data/train.csv'
    VALIDATION_FILE = '../data/validation.csv'
    TEST_FILE = '../data/test.csv'
    ONLINE_FILE = '../data/online_test.txt'
    STOPWORDS_FILE = '../../experiments/stopwords/cn_stopwords.txt'
    WORD2VEC_FILE = '../../experiments/word2vec/sgns.zhihu.word'
    
    # Data field in .csv
    CONTENT_FIELD = 'content'     
    TITLE_FIELD = 'title'

    # Data outputs
    OUTPUT_TEST = '../result/pred_test.csv'      # prediction on test set (labeled)
    OUTPUT_ONLINE = '../result/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'  # prediction on online dataset (no label)
