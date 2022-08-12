import time
import re
import gensim
import jieba
import numpy as np
import pandas as pd
from splitter_sampler import SplitterSampler
from evaluation import MltClsEvaluation, BinClsEvaluation, Evaluation
from modeling import Modeling


class config:
    
    # Data
    DATA_PATH = '../data/all.csv'
    TEST_RATIO = 0.15
    CONTENT_FIELD = 'content'
    LABEL_FIELD = 'class_label'
    STOPWORDS_PATH = '../data/stopwords.txt'
    WORD2VEC_PATH = '../../experiments/word2vec/sgns.sogou.char'

    # Model
    MODEL = 'rf'          # {'rf':RandomForest, 'lr':LogisticRegression ...}, check modeling.py for more details. 
    CV = 0.3              # cross validation 
    MAX_ITER = 4          # iteration for RandomSearch
    
    # Hardware 
    PARALLELISM_TRAINING = 4       # for parallel training
    PARALLELISM_CV = 4             # for cross validation
    
    # Label info
    LABELS = ['中立', '正向', '负向']
    CLASSES = len(LABELS)
    CLS2IDX, IDX2CLS = dict(), dict()
    for i, label in enumerate(LABELS):
        CLS2IDX[label]=i
        IDX2CLS[i]=label
        

def main():
    
    # Load data
    print('Loading data...')
    df = pd.read_csv(config.DATA_PATH)
    df.drop('id',axis=1,inplace=True)
    train_df, test_df = SplitterSampler.split_df(df, config.TEST_RATIO)
    train_df.reset_index(inplace=True,drop=True)
    test_df.reset_index(inplace=True,drop=True)
    print('Training set shape: '+ str(train_df.shape))
    print('Test set shape: '+ str(test_df.shape))
    print('Loading finished.')
    
    # Load stopwords
    stopwords_list = []
    f = open(config.STOPWORDS_PATH)
    lines = f.read()
    stopwords_list = lines.split('\n')
    f.close()
    
    # Load word2vec    
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(config.WORD2VEC_PATH,binary=False)

    def get_text_embedding(x):
        x = ''.join(re.findall('[\u4e00-\u9fa5]', x))                    # remove symbols, numbers and English
        seg_list = jieba.cut(x)                                          # segmentation
        clean_list = [x for x in seg_list if x not in stopwords_list]    # remove stop words
        token_id_list = [word2vec.key_to_index[x] for x in clean_list if x in word2vec.key_to_index.keys()]   # get token ids
        ## calculate embedding
        emb = np.zeros(len(word2vec[0]))
        for idx in token_id_list:
            emb += word2vec[idx]
        emb /= len(token_id_list)
        return emb
    
    # Embedding
    print('Processing dataset...')
    t1 = time.time()
    train_df['embedding'] = train_df[config.CONTENT_FIELD].apply(lambda x: get_text_embedding(x))
    test_df['embedding'] = test_df[config.CONTENT_FIELD].apply(lambda x: get_text_embedding(x))
    X_train = np.stack(train_df['embedding'].to_list(), axis=0)
    X_test = np.stack(test_df['embedding'].to_list(), axis=0)
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    y_train = train_df[config.LABEL_FIELD].map(config.CLS2IDX).tolist()
    y_test = test_df[config.LABEL_FIELD].map(config.CLS2IDX).tolist()
    t2 = time.time()
    print('Processing finished, time consumption in ' + str(t2-t1) + 's.')

    # Modeling
    print('Start modeling...')
    t1 = time.time()
    model = Modeling(X_train, X_test, y_train, y_test, task='mlt', parallelism_training=config.PARALLELISM_TRAINING)
    y_score, y_proba, best_model, best_score, best_params = model.modeling(
        model=config.MODEL, 
        metric='accuracy',
        cv=config.CV,
        hp='auto',
        strategy='random', 
        max_iter=config.MAX_ITER,
        parallelism_cv=config.PARALLELISM_CV, 
        calibration=None
    )
    t2 = time.time()
    print('Modeling finished, time consumption in ' + str(t2-t1) + 's.')

    # Evaluation
    mle = MltClsEvaluation(y_score, y_test, config.LABELS)
    mle.confusion_matrix()
    mle.detailed_metrics()
    
if __name__ == "__main__":
    # execute only if run as a script
    main()

