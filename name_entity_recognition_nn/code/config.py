import datetime

# GPU
CUDA_VISIBLE_DEVICES = '-1'   # disable all GPUs

# Training & Validation 
BATCH_SIZE = 128            # for training, validation and test
ONLINE_BATCH_SIZE = 16      # for online test
EPOCHS = 36
LEARNING_RATE = 0.005
EARLY_STOP_PATIENCE = 4
LR_REDUCE_FACTOR = 0.75
LR_REDUCE_PATIENCE = 1  

# Model structure
MAX_LEN = 128
EMBEDDING_SIZE = 300                   
RECURRENT_DROPOUT_RATE = 0.5
BILSTM_UNITS = 196
DENSE_UNITS = 128

# Model outputs    
ROOT_PATH = '../model/'
TOKENIZER_PATH = ROOT_PATH+'tokenizer.json'
MODEL_PATH = ROOT_PATH+'model.h5'

# Data inputs
TRAINING_FILE = '../data/example.train'   # labeled
VALIDATION_FILE = '../data/example.dev'   # labeled
TEST_FILE = '../data/example.test'        # labeled
ONLINE_FILE = '../data/online_test.txt'   # unlabeled

# Data outputs
REQUIRE_TRANS = True
ONLINE_FILE_PRED = '../result/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.conll'
ONLINE_FILE_PRED_TRANS = '../result/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.txt'

# Log outputs
LOG_DIR = '../log/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
HISTORY_FILE = '../log/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'

# Label info
BASIC_TAGS = ['PER', 'ORG', 'LOC']    # set manually
UNIQUE_TAGS_LEGAL = []
for tag in BASIC_TAGS:
    UNIQUE_TAGS_LEGAL.append('B-'+tag)
    UNIQUE_TAGS_LEGAL.append('I-'+tag)
UNIQUE_TAGS_LEGAL.append('O')
UNIQUE_TAGS = UNIQUE_TAGS_LEGAL + ['PAD']
TAG2ID = dict()
ID2TAG = dict()
for i, item in enumerate(UNIQUE_TAGS):
    TAG2ID[item] = i
    ID2TAG[i] = item
    