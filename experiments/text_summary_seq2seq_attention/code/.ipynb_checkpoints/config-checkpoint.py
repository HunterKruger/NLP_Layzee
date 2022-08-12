'''
Text summarization in Chinese.

CPU
Seq2Seq Model
Word Embedding, not pretrained
Use only the last hidden state of encoder
Attention
Reverse Input
No Teacher Forcing
Greedy search for inference
Bleu & Rouge Score for evalution

'''



import datetime

class config:
    
    # GPU
    CUDA_VISIBLE_DEVICES = '-1'

 
    # Model Hyperparameters
    MAX_LEN_ENCODER = 100
    MAX_LEN_DECODER = 30
    EMBEDDING_DIM = 300
    HIDDEN_UNITS = 500
    BATCH_SIZE = 32
    EPOCHS = 16
    LEARNING_RATE = 0.01
    EARLY_STOP_PATIENCE = 3
    LR_REDUCE_FACTOR = 0.5
    LR_REDUCE_PATIENCE = 1
    RECURRENT_DROPOUT = 0.3
    CLIPNORM = 1

    # Model outputs
    ROOT_PATH = '../model/'
    MODEL_PATH = ROOT_PATH + 'seq2seq.h5'
    ENCODER_PATH = ROOT_PATH + 'encoder.h5'
    DECODER_PATH = ROOT_PATH + 'decoder.h5'

    # Log outputs
    LOG_DIR = "../log/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")                   # log for tensorboard
    HISTORY_FILE = '../log/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv'     # log for training history

    # Data inputs
    ALREADY_SPLIT = False
    TRAIN_SIZE = 4000
    TEST_SIZE = 100
    VALIDATION_SIZE = 500
    REVERSE = True
    NUM_WORDS = 20000
    ORIGIN_FILE = '../origin_data/lcsts_data.json'
    ALL_FILE = '../data/all.csv'
    TRAIN_FILE = '../data/train.csv'
    VALIDATION_FILE = '../data/validation.csv'
    TEST_FILE = '../data/test.csv'
    ONLINE_FILE = '../data/online_test.txt'
    TOKENIZER_PATH = '../model/tokenizer.json'
    STOPWORDS_FILE = '../../experiments/stopwords/cn_stopwords.txt'
    WORD2VEC_FILE = '../../experiments/word2vec/sgns.zhihu.word'
    
    # Data field in .csv
    CONTENT_FIELD = 'content'     
    TITLE_FIELD = 'title'

    # Data outputs
    OUTPUT_TEST = '../result/pred_test.csv'      # prediction on test set (labeled)
    OUTPUT_ONLINE = '../result/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'  # prediction on online dataset (no label)
