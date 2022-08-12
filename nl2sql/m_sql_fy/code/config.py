import datetime

# GPU/CPU specification
CUDA_VISIBLE_DEVICES = '0,1,2,3'                    # specifiy GPU ids (use nvidia-smi) in dp mode
GPUS = len(CUDA_VISIBLE_DEVICES.split(','))         # count of GPUs
ONLY_CPU = False                                    # only use CPU for inference 
if ONLY_CPU:
    CUDA_VISIBLE_DEVICES = ''

# CPU multi-threading, for dataloader
NUM_WORKERS = 4
if not ONLY_CPU:
    NUM_WORKERS *= GPUS  

# Data inputs
train_table_file = '../../TableQA-master/train/train.tables.json'
train_data_file = '../../TableQA-master/train/train.json'
val_table_file = '../../TableQA-master/val/val.tables.json'
val_data_file = '../../TableQA-master/val/val.json'
test_table_file = '../../TableQA-master/test/test.tables.json'
test_data_file = '../../TableQA-master/test/test.json'

# Model inputs
BASE_MODEL_PATH = '../../../experiments/model/chinese_wwm_ext_pytorch'

# Model outputs
ROOT_MODEL_PATH = '../model/'   

# Training & Validation
LOG_INTERVAL = 1
MAX_LEN = 128
BATCH_SIZE = 32      # on single GPU/CPU
if not ONLY_CPU:
    BATCH_SIZE *= GPUS
LR = 2e-6
EPOCHS = 1

# Labels
def label_encoder(label_list):
    label2id = dict()
    id2label = dict()
    for i, label in enumerate(label_list):
        label2id[label] = i
        id2label[i] = label
    return label2id, id2label

#S_num_labels = [1,2,3]
S_num_labels = [1,2]
#W_num_op_labels = ['NULL-1','OR-2','AND-2','OR-3','AND-3','OR-4','AND-4']
W_num_op_labels = ['NULL-1','OR-2','AND-2']
W_col_val_labels = [0,1]

S_num_label2id, S_num_id2label = label_encoder(S_num_labels)
W_num_op_label2id, W_num_op_id2label = label_encoder(W_num_op_labels)

# Log outputs
TENSORBOARD_FILE = '../tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")       # log for tensorboard
TENSORBOARD_TIME = 100000                                                                      # tensorboard flush seconds, choose a value which is longer than the training process
OUTPUT_CONFIG = '../config/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.py'       # save config for each training
