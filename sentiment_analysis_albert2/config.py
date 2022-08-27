FILE_NAME = 'ABT_S_0822'

ROOT_PATH = '../experiments/' + FILE_NAME + '/'

# CPU multi-threading, for dataloader
NUM_WORKERS = 8

# For spliting dataset
RANDOM_STATE = 1           
TEST_SIZE = 0.1             
VALIDATION_SIZE = 0.15   

# Training & Validation 
BATCH_SIZE = 128           # for training, validation and test
LOG_INTERVAL = 1           # step intervals for displaying train loss and metrics 
MAX_LEN = 256              # Maximum input sequence length
EPOCHS = 5                 # epochs
PATIENCE = 1               # for early stopping
DROPOUT_RATE = 0.3
FACTOR = 0.5               # for ReduceLRonPlateau
REDUCE_PAT = 1             # for ReduceLRonPlateau

# Adam + WeightDecay
LR = 2e-5              # Initial learning rate (just after warmup)
WEIGHT_DECAY = 0.005   # weight:= weight*(1-decay_rate)
CLIP_NORM = 1.0        # gradient clipping with normalization

# Model inputs
BASE_MODEL_PATH = '../pretrained_models/voidful_albert_chinese_small'

# Model outputs
LABEL_ENCODER_PATH = ROOT_PATH + 'label_encoder.sav'
CKPT_PATH = ROOT_PATH + 'model.hdf5'

# Log outputs
TENSORBOARD_DIR = '/home/powerop/work/tf_log/' + FILE_NAME    # log for tensorboard
CONFIG_FILE = ROOT_PATH + 'config.py'                         # log for config.py

# Data inputs
CONTENT_FIELD = 'writingcontent'               # field of text in input file
LABEL_FIELD = 'label'                          # field of label in input file
INPUT_FILE = '../data/train_20220822_nlp.csv'

# Data outputs
TRAIN_FILE      = ROOT_PATH + 'train.csv'
VALIDATION_FILE = ROOT_PATH + 'validation.csv'
TEST_FILE       = ROOT_PATH + 'test.csv'
OUTPUT_TEST     = ROOT_PATH + 'pred.csv'       # prediction on test set (labeled)
