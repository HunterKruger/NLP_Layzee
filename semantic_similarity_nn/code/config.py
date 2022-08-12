import datetime
    
# Modeling Strategy 
USE_RNN = False          # use CNN as base model if False
FAKE_SIAMESE = False     # base models share their params if True

# CPU
CPUS = 4                 # multi-threading for data processing
MAX_QUEUE_SIZE = 10

# GPU
if USE_RNN:
    CUDA_VISIBLE_DEVICES = '-1'        # specifiy GPU ids (use nvidia-smi)
    GPUS = len(CUDA_VISIBLE_DEVICES.split(','))               
else:
    CUDA_VISIBLE_DEVICES = '0,1,2,3'   # specifiy GPU ids (use nvidia-smi)

# Dataset
TEST_SIZE = 0.15           # split origin data set into (big) training set & test set
VALIDATION_SIZE = 0.2      # split (big) training set into training set & validation set
RANDOM_STATE = 1           # for spliting dataset
UNDER_SAMPLING = True

# Training & Validation 
MAX_LEN = 32
BATCH_SIZE = 128            # for training, validation and test
ONLINE_BATCH_SIZE = 16      # for online test
EPOCHS = 16
LEARNING_RATE = 0.005
EARLY_STOP_PATIENCE = 4
LR_REDUCE_FACTOR = 0.75
LR_REDUCE_PATIENCE = 1  
MARGIN = 1                  # for contractive loss
    
# Model Structure
BILSTM_UNITS = 96                           # specify it only when USE_RNN=True
FILTERS = [64, 64, 64, 64, 64]              # specify it only when USE_RNN=False
KERNEL_SIZE = [32, 16, 8, 4, 2]             # specify it only when USE_RNN=False
STRIDES = [16, 8, 4, 2, 1]                  # specify it only when USE_RNN=False

# Model outputs
ROOT_PATH = '../model/'        
if USE_RNN:
    MODEL_PATH = ROOT_PATH+'rnn_siamese.h5'    
else:
    MODEL_PATH = ROOT_PATH+'cnn_siamese.h5' 
    
# Log outputs
LOG_DIR = "../log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")                   # log for tensorboard
HISTORY_FILE = '../log/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'   # log for training history

# Data inputs
ALREADY_SPLIT = False                                           # data set has already been split
ALL_FILE = '../data/all.csv'                                  # labeled samples, will be split into training set & validation set & test set
TRAIN_FILE = '../data/train.csv'
VALIDATION_FILE = '../data/validation.csv'
TEST_FILE = '../data/test.csv'
ONLINE_FILE = '../data/online_test.txt'                         # unlabeled samples, for batch inference
SENTENCE_FIELD = 's1'
SENTENCE_FIELD2 = 's2'
LABEL_FIELD = 'label'                              
STOPWORDS_FILE = '../../experiments/stopwords/cn_stopwords.txt' 
WORD2VEC_FILE = '../../experiments/word2vec/sgns.zhihu.word'    

# Data outputs
OUTPUT_TEST = '../result/pred_test.csv' # prediction on test set (labeled)
OUTPUT_ONLINE = '../result/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'  # prediction on online dataset 
