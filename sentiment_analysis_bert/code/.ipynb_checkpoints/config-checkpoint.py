import datetime

class config:
     
    # CPU
    CPUS = 4                 # multi-threading for data processing
    MAX_QUEUE_SIZE = 10
    
    # GPU
    CUDA_VISIBLE_DEVICES = '0,1,2,3'   # specifiy GPU ids (use nvidia-smi)
    GPUS = len(CUDA_VISIBLE_DEVICES.split(','))               

    # Training & Validation 
    TEST_SIZE = 0.15           # split origin data set into (big) training set & test set
    VALIDATION_SIZE = 0.2      # split (big) training set into training set & validation set
    RANDOM_STATE = 1           # for spliting dataset
    MAX_LEN = 196
    BATCH_SIZE = 64            # for training, validation and test
    ONLINE_BATCH_SIZE = 16     # for online test
    EPOCHS = 6                 

    # PHASE1: Adam + Warmup + PolynomialDecay within mini-batch
    INIT_LR = 3e-5             # Inital learning rate (just after warmup)
    NUM_WARMUP_STEPS = 32     # Number of warmup steps
    MIN_LR_RATIO = 0.1         # The final learning rate at the end of the linear decay will be INIT_LR * MIN_LR_RATIO
    WEIGHT_DECAY_RATE = 0.75   # The weight decay to use
    POWER = 0.5                # The power to use for PolynomialDecay
    
    # PHASE2: SGD + Momentum + Nesterov + ExponentialDecay within mini-batch
    SGD_LR = 5e-6
    SGD_MOMENTUM = 0.9
    SGD_NESTEROV = True
    SGD_DECAY_STEPS = 1024
    SGD_DECAY_RATE = 0.98
       
    # Model structure 
    DROPOUT_RATE = 0.3
    FREEZE_BERT_LAYERS_LIST = [0,1]      # choose BERT layers to freeze from 0 ~ 11, or use -1, -2 ...;  for PHASE1
    FREEZE_BERT_LAYERS_LIST2 = []        # for PHASE2
    FREEZE_WHOLE_BERT = False            # freeze all BERT params; lower priority than FREEZE_BERT_LAYERS_LIST
    HIDDEN_STATE_LIST = [-1,-2,-3,-4]             # choose BERT layers to get their hidden states, then concatenate them; 

    # Model inputs
    BASE_MODEL_PATH = '../../experiments/model/bert-base-chinese'

    # Model outputs
    ROOT_PATH = '../model/'                               # for PHASE1
    ROOT_PATH2 = '../model2/'                             # for PHASE2
    CHECKPOINT_PATH = ROOT_PATH+'cp_{epoch:04d}.ckpt'     # for PHASE1
    CHECKPOINT_PATH2 = ROOT_PATH2+'cp_{epoch:04d}.ckpt'   # for PHASE2
    MODEL_PATH = '../model/best_model.h5'                 # optional, not used
    
    # Log outputs
    LOG_DIR="../log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")                   # log for tensorboard
    HISTORY_FILE = '../log/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'   # log for training history

    # Data inputs
    ALREADY_SPLIT = True                   # data set has already been split
    INPUT_FILE = '../data/all.csv'         # labeled samples, will be split into training set & validation set & test set
    TRAIN_FILE = '../data/train.csv'
    VALIDATION_FILE = '../data/validation.csv'
    TEST_FILE = '../data/test.csv'
    ONLINE_FILE = '../data/online_test.txt' # unlabeled samples
    CONTENT_FIELD = 'content'               # field of text in input file
    LABEL_FIELD = 'class_label'             # field of label in input file

    # Data outputs
    OUTPUT_TEST = '../result/pred_test.csv' # prediction on test set (labeled)
    OUTPUT_ONLINE = '../result/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'  # prediction on online dataset 

    # Label info
    LABELS = ['中立', '正向', '负向']
    CLASSES = len(LABELS)
    CLS2IDX, IDX2CLS = dict(), dict()
    for i, label in enumerate(LABELS):
        CLS2IDX[label]=i
        IDX2CLS[i]=label

