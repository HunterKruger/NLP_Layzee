import datetime

### CPU
CPUS = 4                 # for multi-threading
MAX_QUEUE_SIZE = 10

### GPU
CUDA_VISIBLE_DEVICES = '0,1,2,3'   # specifiy GPU ids (use nvidia-smi)
GPUS = len(CUDA_VISIBLE_DEVICES.split(','))

### Data spliting
TEST_SIZE = 0.15         # split origin data set into (big) training set & test set
VALIDATION_SIZE = 0.2    # split (big) training set into training set & validation set
RANDOM_STATE = 1         # for spliting dataset
UNDER_SAMPLING = True

### Model Hyperparameters
MAX_LEN = 64
MAX_LEN_SIAMESE = 32
EPOCHS = 4
BATCH_SIZE = 64           # for train, val and test
ONLINE_BATCH_SIZE = 16    # for online test

### Adam + Warmup + Decay
INIT_LR = 3e-5             # Inital learning rate (just after warmup)
NUM_WARMUP_STEPS = 96      # Number of warmup steps
MIN_LR_RATIO = 0.1         # The final learning rate at the end of the linear decay will be INIT_LR * MIN_LR_RATIO
WEIGHT_DECAY_RATE = 0.75   # The weight decay to use
POWER = 0.5                # The power to use for PolynomialDecay

### Model structure
DENSE_UNITS = 64                 # for non-siamese case
DROPOUT_RATE = 0.3               # for siamese case
FREEZE_BERT_LAYERS_LIST = []     # choose BERT layers to freeze from 0 ~ 11, or use -1, -2 ...
FREEZE_WHOLE_BERT = False        # freeze all BERT params; lower priority than FREEZE_BERT_LAYERS_LIST
USE_SIAMESE = False              

### Model inputs
BASE_MODEL_PATH = '../../experiments/model/bert-base-chinese'

### Model outputs
ROOT_PATH = '../model/'                     
CHECKPOINT_PATH = ROOT_PATH + 'cp_{epoch:04d}.ckpt'
ROOT_PATH2 = '../model_siamese/'                      
CHECKPOINT_PATH2 = ROOT_PATH2 + 'cp_{epoch:04d}.ckpt'

### Log outputs
LOG_DIR="../log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
HISTORY_FILE = '../log/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'

### Data inputs
ALREADY_SPLIT = False                        # set to False if only provides INPUT_FILE
INPUT_FILE = '../data/all.csv'               # full scale dataset with labeled samples, will be split into training set & validation set & test set
TRAIN_FILE = '../data/train.csv'             
VALIDATION_FILE = '../data/validation.csv'
TEST_FILE = '../data/test.csv'
ONLINE_TEST_FILE = '../data/online_test.txt'
SENTENCE_FIELD = 's1'
SENTENCE_FIELD2 = 's2'
LABEL_FIELD = 'label'

### Data outputs
OUTPUT_TEST = '../result/pred_test.csv'      # prediction on test set (labeled)
OUTPUT_ONLINE = '../result/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'  # prediction on online dataset (no label)
    