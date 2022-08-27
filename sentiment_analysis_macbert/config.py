FILE_NAME = 'J001'

import datetime

ROOT_PATH = '../experiments/' + FILE_NAME + '/'

# CPU
CPUS = 8                    # multi-threading for data processing
MAX_QUEUE_SIZE = 20

# Training & Validation 
RANDOM_STATE = 1            # for spliting dataset
TEST_SIZE = 0.1             
VALIDATION_SIZE = 0.15       
BATCH_SIZE = 128            # for training, validation and test
EPOCHS = 5      
MAX_LEN = 256

# Optimizer
LR = 2e-5                   # Initial learning rate 

# Focal loss
GAMMA = 2                   # set GAMMA to 0 -> CE loss

# Callbacks
ES_PAT = 1
RD_LR_FAC = 0.5
RD_LR_PAT = 1
    
# Model structure 
DROPOUT_RATE = 0.3
FREEZE_BERT_LAYERS_LIST = []              # choose BERT layers to freeze from 0 ~ 11, or use -1, -2 ...
FREEZE_WHOLE_BERT = False                 # freeze all BERT params; lower priority than FREEZE_BERT_LAYERS_LIST
HIDDEN_STATE_LIST = [-1, -2, -11, -12]    # choose BERT layers to get their hidden states pooled then concatenated

# Model inputs
BASE_MODEL_PATH = '../pretrained_models/macbert_chinese_base/'

# Model outputs
LABEL_ENCODER_PATH = ROOT_PATH + 'label_encoder.sav'
CKPT_PATH = ROOT_PATH + 'model.hdf5'

# Log outputs
TENSORBOARD_DIR = '/home/powerop/work/tf_log/' + FILE_NAME    # log for tensorboard
CONFIG_FILE = ROOT_PATH + 'config.py'                         # log for config.py

# Data inputs
INPUT_FILE = '../data/train_hly_nlp.csv'
CONTENT_FIELD = 'writingcontent'                      
LABEL_FIELD = 'label'                  

# Data outputs
TRAIN_FILE      = ROOT_PATH + 'train.csv'
VALIDATION_FILE = ROOT_PATH + 'validation.csv'
TEST_FILE       = ROOT_PATH + 'test.csv'
OUTPUT_TEST     = ROOT_PATH + 'pred.csv'       # prediction on test set (labeled)


