"""
Created by FENG YUAN on 2021/10/6
"""

import datetime

# GPU/CPU specification
CUDA_VISIBLE_DEVICES = '4,5,6,7,8,9,10,11'            # specifiy GPU ids (use nvidia-smi) in dp mode
GPUS = len(CUDA_VISIBLE_DEVICES.split(','))         # count of GPUs
ONLY_CPU = False                                    # only use CPU for inference
if ONLY_CPU:
    CUDA_VISIBLE_DEVICES = ''

# CPU multi-threading, for dataloader
NUM_WORKERS = 4
if not ONLY_CPU:
    NUM_WORKERS *= GPUS

# Data inputs
TRAIN_TABLE_FILE = '../../TableQA-master/train/train.tables.json'
TRAIN_DATA_FILE = '../../TableQA-master/train/train.json'
VAL_TABLE_FILE = '../../TableQA-master/val/val.tables.json'
VAL_DATA_FILE = '../../TableQA-master/val/val.json'
TEST_TABLE_FILE = '../../TableQA-master/test/test.tables.json'
TEST_DATA_FILE = '../../TableQA-master/test/test.json'

# Model inputs
BASE_MODEL_PATH = '../../../NLP/experiments/model/albert_chinese_base'   #chinese_wwm_pytorch

# Model outputs
ROOT_MODEL_PATH = '../model/'

# Training & Validation
LOG_INTERVAL = 1
MAX_LEN = 160
MAX_HEADERS = 30
BATCH_SIZE = 32  # on single GPU/CPU
if not ONLY_CPU:
    BATCH_SIZE *= GPUS
LR = 1e-5
EPOCHS = 50
PATIENCE = 2

# Labels
def label_encoder(label_list):
    label2id = dict()
    id2label = dict()
    for i, label in enumerate(label_list):
        label2id[label] = i
        id2label[i] = label
    return label2id, id2label


COND_OP_DICT = {0: ">", 1: "<", 2: "==", 3: "!="}                           # 4 for None
SEL_AGG_DICT = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}  # 6 for None
COND_CONN_OP_DICT = {0: "NULL", 1: "AND", 2: "OR"}

# Log outputs
OUTPUT_CONFIG = '../config/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.py'    # save config for each training
TENSORBOARD_FILE = '../tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")    # log for tensorboard
TENSORBOARD_TIME = 100000                                                                   # tensorboard flush seconds, choose a value which is longer than the training process

# Inference results
TASK1_RESULT = '../prediction/task1_result.json'
TASK2_RESULT = '../prediction/task2_result.json'