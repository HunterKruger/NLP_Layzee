import datetime

# GPU
CUDA_VISIBLE_DEVICES = '0,1,2,3'   # specifiy GPU ids (use nvidia-smi)
GPUS = len(CUDA_VISIBLE_DEVICES.split(','))    

# CPU
CPUS = GPUS*4               # multi-threading for data processing
MAX_QUEUE_SIZE = 10

# Training & Validation 
TEST_SIZE = 0.15            # split origin data set into (big) training set & test set
VALIDATION_SIZE = 0.2       # split (big) training set into training set & validation set
RANDOM_STATE = 1            # for spliting dataset
BATCH_SIZE = 64             # for training, validation and test
ONLINE_BATCH_SIZE = 16      # for online test
EPOCHS = 3        
MAX_LEN = 196

# PHASE1: Adam + WeightDecay + LR Warmup + LR PolynomialDecay
ADAM_LR = 3e-5              # Inital learning rate (just after warmup)
ADAM_WARMUP_STEPS = 40      # Number of warmup steps
ADAM_MIN_LR_RATIO = 0.1     # The final learning rate at the end of the linear decay will be INIT_LR * MIN_LR_RATIO
ADAM_DECAY_RATE = 0.001     # The weight decay to use
ADAM_POWER = 0.5            # The power to use for PolynomialDecay

# PHASE2: SGD + Momentum + Nesterov + ExponentialDecay 
SGD_LR = 5e-6
SGD_MOMENTUM = 0.9
SGD_NESTEROV = True
SGD_DECAY_STEPS = 10        # decay every SGD_DECAY_STEPS with SGD_DECAY_RATE
SGD_DECAY_RATE = 0.95
    
# Model structure 
DROPOUT_RATE = 0.3
FREEZE_BERT_LAYERS_LIST = []         # choose BERT layers to freeze from 0 ~ 11, or use -1, -2 ...;  for PHASE1
FREEZE_WHOLE_BERT = False            # freeze all BERT params; lower priority than FREEZE_BERT_LAYERS_LIST
HIDDEN_STATE_LIST = [-1,-2,-3,-4]    # choose BERT layers to get their hidden states pooled then concatenated

# Model inputs
BASE_MODEL_PATH = '../../experiments/model/bert-base-chinese'

# Model outputs
ROOT_PATH = '../model/'                              

# Log outputs
TENSORBOARD_DIR = '../tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")    # log for tensorboard
HISTORY_FILE = '../log/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'      # log for training history
CONFIG_FILE = '../config/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.py'     # log for config.py

# Data inputs
ALREADY_SPLIT = True                        # data set has already been split
INPUT_FILE = '../data/all.csv'              # labeled samples, will be split into training set & validation set & test set
TRAIN_FILE = '../data/train.csv'
VALIDATION_FILE = '../data/validation.csv'
TEST_FILE = '../data/test.csv'
ONLINE_FILE = '../data/online_test.txt'     # unlabeled samples
CONTENT_FIELD = 'content'                   # field of text in input file
LABEL_FIELD = 'class_label'                 # field of label in input file

# Data outputs
OUTPUT_TEST = '../test_result/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'        # prediction on test set (labeled)
OUTPUT_ONLINE = '../online_result/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'    # prediction on online dataset 

# Label info
LABELS = ['中立', '正向', '负向']
CLASSES = len(LABELS)
CLS2IDX, IDX2CLS = dict(), dict()
for i, label in enumerate(LABELS):
    CLS2IDX[label]=i
    IDX2CLS[i]=label

