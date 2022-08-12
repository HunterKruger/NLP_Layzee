import datetime

# GPU
CUDA_VISIBLE_DEVICES = '0,1,2,3'         # specifiy GPU ids (use nvidia-smi) in dp mode
GPUS = len(CUDA_VISIBLE_DEVICES.split(','))   

# CPU multi-threading
NUM_WORKERS = 4*GPUS  # for dataloader 

# Training & Validation 
TEST_SIZE = 0.15           # split origin data set into (big) training set & test set
VALIDATION_SIZE = 0.2      # split (big) training set into training set & validation set
RANDOM_STATE = 1           # for spliting dataset
MAX_LEN = 196
BATCH_SIZE = 256           # for training, validation and test
                                # in ddp model, each gpu gets BATCH_SIZE for each step
                                # in dp model, each gpu gets BATCH_SIZE/GPUS for each step
ONLINE_BATCH_SIZE = 16     # for online inference
EPOCHS = 4

# Model structure if use model.py
HIDDEN_STATE_LIST = [-1, -2, -3, -4]   # 13 layers, embedding + 12 encoder layers, index from 0 to 12
DROPOUT_RATE = 0.3
DENSE_UNITS = 32

# Adam/SGD + LinearWarmup + WeightDecay
LR = 8e-6              # Initial learning rate (just after warmup)
WARMUP_STEPS = 10      # Number of warmup steps
WEIGHT_DECAY = 0.005   # weight:= weight*(1-decay_rate)
MOMENTUM = 0.9         # for SGD
NESTEROV = True        # for SGD

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
HISTORY_FILE = '../log/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'   # log for training history

# Data outputs
OUTPUT_TEST = '../result/pred_test.csv'                                                     # prediction on test set (labeled)
OUTPUT_ONLINE = '../result/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'   # prediction on online dataset 
OUTPUT_CONFIG = '../config/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.py'

# Label info
LABELS = ['中立', '正向', '负向']
CLASSES = len(LABELS)
CLS2IDX, IDX2CLS = dict(), dict()
for i, label in enumerate(LABELS):
    CLS2IDX[label]=i
    IDX2CLS[i]=label

