import datetime

# GPU/CPU specification
CUDA_VISIBLE_DEVICES = '0,1,2,3'                    # specifiy GPU ids (use nvidia-smi) in dp mode
GPUS = len(CUDA_VISIBLE_DEVICES.split(','))         # count of GPUs
ONLY_CPU = False                                     # only use CPU for inference 
if ONLY_CPU:
    CUDA_VISIBLE_DEVICES = ''

# CPU multi-threading, for dataloader
NUM_WORKERS = 4
if not ONLY_CPU:
    NUM_WORKERS *= GPUS  

# Training & Validation 
TEST_SIZE = 0.15           # split origin data set into (big) training set & test set
VALIDATION_SIZE = 0.2      # split (big) training set into training set & validation set
RANDOM_STATE = 1           # for spliting dataset
BATCH_SIZE = 64            # for training, validation and test
                                # in ddp model, each gpu gets BATCH_SIZE for each step
                                # in dp model, each gpu gets BATCH_SIZE/GPUS for each step
ONLINE_BATCH_SIZE = 16     # for online batch inference
LOG_INTERVAL = 1           # step intervals for displaying train loss and metrics 
MAX_LEN = 196              # Maximun input sequence length
EPOCHS = 10                # epochs

# Model structure if use model.py
HIDDEN_STATE_LIST = [-1, -2, -3, -4]   # 13 layers, embedding + 12 encoder layers, index from 0 to 12
DROPOUT_RATE = 0.3
DENSE_UNITS = 32

# Adam/SGD + LinearWarmup + WeightDecay
LR = 3e-5              # Initial learning rate (just after warmup)
WARMUP_STEPS = 10      # Number of warmup steps
LR_END = 1e-5          # LR at the end of LR decay
POWER = 0.5            # The power to use for PolynomialDecay
SCHEDULER_STEPS = 300  # Steps of lr scheduler
WEIGHT_DECAY = 0.005   # weight:= weight*(1-decay_rate)
MOMENTUM = 0.9         # for SGD
NESTEROV = True        # for SGD
CLIP_NORM = 1.0        # gradient clipping with normalization

# Model inputs
BASE_MODEL_PATH = '../../experiments/model/albert_chinese_base'

# Data inputs
ALREADY_SPLIT = True                    # data set has already been split
INPUT_FILE = '../data/all.csv'          # labeled samples, will be split into training set & validation set & test set
TRAIN_FILE = '../data/train.csv'
VALIDATION_FILE = '../data/validation.csv'
TEST_FILE = '../data/test.csv'
ONLINE_FILE = '../data/online_test.txt' # unlabeled samples
CONTENT_FIELD = 'content'               # field of text in input file
LABEL_FIELD = 'class_label'             # field of label in input file

# Model outputs
ROOT_PATH = '../model/'                               

# Log outputs
TENSORBOARD_FILE = '../tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # log for tensorboard
TENSORBOARD_TIME = 10000                                                                  # tensorboard flush seconds, choose a value which is longer than the training process

# Data outputs
OUTPUT_TEST = '../test_result/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'       # prediction on test set (labeled)
OUTPUT_ONLINE = '../online_result/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'   # prediction on online dataset 
OUTPUT_CONFIG = '../config/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.py'           # save config for each training

# Label info
LABELS = ['中立', '正向', '负向']    # specify all class labels in dataset
CLASSES = len(LABELS)
CLS2IDX, IDX2CLS = dict(), dict()
for i, label in enumerate(LABELS):
    CLS2IDX[label]=i
    IDX2CLS[i]=label

