import datetime


# CPU
CPUS = 4                 # for multi-threading
MAX_QUEUE_SIZE = 10

# GPU
CUDA_VISIBLE_DEVICES = '0,1,2,3'   # specifiy GPU ids (use nvidia-smi)
GPUS = len(CUDA_VISIBLE_DEVICES.split(','))

# Model hyper-parameters
MAX_LEN = 128
BATCH_SIZE = 64
ONLINE_BATCH_SIZE = 32
EPOCHS = 4

# Adam + Warmup + Decay
INIT_LR = 3e-5             # Inital learning rate (just after warmup)
NUM_WARMUP_STEPS = 128     # Number of warmup steps
MIN_LR_RATIO = 0.1         # The final learning rate at the end of the linear decay will be INIT_LR * MIN_LR_RATIO
WEIGHT_DECAY_RATE = 0.75   # The weight decay to use
POWER = 0.5                # The power to use for PolynomialDecay

# Model structure
DROPOUT_RATE = 0.3
RECURRENT_DROPOUT_RATE = 0.5
BILSTM_UNITS = 256
DENSE_UNITS = 128
FREEZE_BERT_LAYERS_LIST = []  # choose BERT layers to freeze from 0 ~ 11, or use -1, -2 ...
FREEZE_WHOLE_BERT = False     # freeze all BERT params; lower priority than FREEZE_BERT_LAYERS_LIST

# Model inputs
BASE_MODEL_PATH = '../../experiments/model/bert-base-chinese'

# Model outputs
CHECKPOINT_PATH = '../model/cp_{epoch:04d}.ckpt'
MODEL_PATH = '../model/best_model.h5'    # optional, not used
ROOT_PATH = '../model/'                  # optional, not used

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
UNIQUE_TAGS = UNIQUE_TAGS_LEGAL + ['PAD', 'SEP', 'CLS']
TAG2ID = dict()
ID2TAG = dict()
for i, item in enumerate(UNIQUE_TAGS):
    TAG2ID[item] = i
    ID2TAG[i] = item
    