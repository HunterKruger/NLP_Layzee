import datetime

class config:
    
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
    
    ### Model Hyperparameters
    MAX_LEN = 128
    EPOCHS = 4
    BATCH_SIZE = 96          # for train, val and test
    ONLINE_BATCH_SIZE = 4    # for online test
    
    ### Adam + Warmup + Decay
    INIT_LR = 3e-5             # Inital learning rate (just after warmup)
    NUM_WARMUP_STEPS = 96      # Number of warmup steps
    MIN_LR_RATIO = 0.1         # The final learning rate at the end of the linear decay will be INIT_LR * MIN_LR_RATIO
    WEIGHT_DECAY_RATE = 0.75   # The weight decay to use
    POWER = 0.5                # The power to use for PolynomialDecay
    
    ### Model structure
    DENSE_UNITS = 64
    DROPOUT_RATE = 0.3
    FREEZE_BERT_LAYERS_LIST = []     # choose BERT layers to freeze from 0 ~ 11, or use -1, -2 ...
    FREEZE_WHOLE_BERT = False        # freeze all BERT params; lower priority than FREEZE_BERT_LAYERS_LIST

    ### Model inputs
    BASE_MODEL_PATH = '../../../JSTelecom/experiments/model/bert-base-chinese'
    
    ### Model outputs
    ROOT_PATH = '../model/'                      # optional, not used
    MODEL_PATH = '../model/best_model.h5'        # optional, not used    
    CHECKPOINT_PATH = '../model/cp_{epoch:04d}.ckpt'

    ### Log outputs
    LOG_DIR="../log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    HISTORY_FILE = '../log/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
    
    ### Data inputs
    ALREADY_SPLIT = True                         # set to False if only provides INPUT_FILE
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
        