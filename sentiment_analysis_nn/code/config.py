import datetime


# Modeling Strategy 
USE_RNN = True          # use CNN if False

# CPU
CPUS = 4                 # multi-threading for data processing
MAX_QUEUE_SIZE = 10

# GPU
if USE_RNN:
    CUDA_VISIBLE_DEVICES = '-1'   # specifiy GPU ids (use nvidia-smi)
    GPUS = len(CUDA_VISIBLE_DEVICES.split(','))               
else:
    CUDA_VISIBLE_DEVICES = '0,1,2,3'   # specifiy GPU ids (use nvidia-smi)

# Dataset
TEST_SIZE = 0.15           # split origin data set into (big) training set & test set
VALIDATION_SIZE = 0.2      # split (big) training set into training set & validation set
RANDOM_STATE = 1           # for spliting dataset

# Training & Validation 
MAX_LEN = 256
BATCH_SIZE = 64            # for training, validation and test
ONLINE_BATCH_SIZE = 16     # for online test
EPOCHS = 16
LEARNING_RATE = 0.01
EARLY_STOP_PATIENCE = 3
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 1  
    
# Model Structure
BILSTM_UNITS = 96                        # specify it only when USE_RNN=True
FILTERS = [64, 64, 64, 64]               # specify it only when USE_RNN=False
KERNEL_SIZE = [32, 16, 8, 4]             # specify it only when USE_RNN=False
STRIDES = [16, 8, 4, 2]                  # specify it only when USE_RNN=False

# Model outputs
ROOT_PATH = '../model/'        
if USE_RNN:
    MODEL_PATH = ROOT_PATH+'rnn.h5'    
else:
    MODEL_PATH = ROOT_PATH+'cnn.h5' 
    
# Log outputs
LOG_DIR="../log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")                   # log for tensorboard
HISTORY_FILE = '../log/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'   # log for training history

# Data inputs
ALREADY_SPLIT = False                                           # data set has already been split
INPUT_FILE = '../data/all.csv'                                  # labeled samples, will be split into training set & validation set & test set
TRAIN_FILE = '../data/train.csv'
VALIDATION_FILE = '../data/validation.csv'
TEST_FILE = '../data/test.csv'
ONLINE_FILE = '../data/online_test.txt'                         # unlabeled samples, for batch inference
CONTENT_FIELD = 'content'                                       # field of text in input file
LABEL_FIELD = 'class_label'                                     # field of label in input file
STOPWORDS_FILE = '../../experiments/stopwords/cn_stopwords.txt' 
WORD2VEC_FILE = '../../experiments/word2vec/sgns.zhihu.word'    

# Data outputs
OUTPUT_TEST = '../result/pred_test.csv' # prediction on test set (labeled)
OUTPUT_ONLINE = '../result/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'  # prediction on online dataset 

# Label info
LABELS = ['中立', '正向', '负向']
CLASSES = len(LABELS)
CLS2IDX, IDX2CLS = dict(), dict()
for i, label in enumerate(LABELS):
    CLS2IDX[label]=i
    IDX2CLS[i]=label