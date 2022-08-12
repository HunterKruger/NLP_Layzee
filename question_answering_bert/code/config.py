import datetime


# CPU
CPUS = 4                 # for multi-threading
MAX_QUEUE_SIZE = 10

# GPU
CUDA_VISIBLE_DEVICES = '0,1,2,3'
GPUS = len(CUDA_VISIBLE_DEVICES.split(','))              

# Model Hyperparameters
MAX_LEN = 256
BATCH_SIZE = 64
EPOCHS = 4

# Adam + Warmup + Decay
INIT_LR = 3e-5             # Inital learning rate (just after warmup)
NUM_WARMUP_STEPS = 96      # Number of warmup steps
MIN_LR_RATIO = 0.1         # The final learning rate at the end of the linear decay will be INIT_LR * MIN_LR_RATIO
WEIGHT_DECAY_RATE = 0.75   # The weight decay to use
POWER = 0.5                # The power to use for PolynomialDecay

# Model structure
FREEZE_BERT_LAYERS_LIST = []   # choose BERT layers to freeze from 0 ~ 11, or use -1, -2 ...
FREEZE_WHOLE_BERT = False      # freeze all BERT params; lower priority than FREEZE_BERT_LAYERS_LIST

# Model inputs
BASE_MODEL_PATH = '../../experiments/model/bert-base-chinese'

# Model outputs
ROOT_PATH = '../model/'
CHECKPOINT_PATH = '../model/cp_{epoch:04d}.ckpt'

# Log outputs
LOG_DIR="../log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")                   # log for tensorboard
HISTORY_FILE = '../log/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'   # log for training history

# Data inputs
ALREADY_SPLIT = True
ALL_FILE = '../data/all.csv'
TRAIN_FILE = '../data/train.csv'
VALIDATION_FILE = '../data/validation.csv'
TEST_FILE = '../data/test.csv'
ONLINE_FILE = '../data/online_test.txt'
TRAIN_SIZE = 65536
VALIDATION_SIZE = 16384
TEST_SIZE = 16384

# Data field in .csv
QUESTION_FIELD = 'question'     
DOCUMENT_FIELD = 'document'
START_FIELD = 'start'
END_FIELD = 'end'

# Data outputs
OUTPUT_TEST = '../result/pred_test.csv'      # prediction on test set (labeled)
OUTPUT_ONLINE = '../result/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'  # prediction on online dataset (no label)
