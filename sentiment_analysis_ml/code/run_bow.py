import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from splitter_sampler import SplitterSampler
from evaluation import MltClsEvaluation, BinClsEvaluation, Evaluation
from modeling import Modeling
from sklearn.naive_bayes import MultinomialNB
import time

def clean(x):
    x = ''.join(re.findall('[\u4e00-\u9fa5]', x))                    # remove symbols, numbers and English
    seg_list = jieba.cut(x)                                          # segmentation
    token_list = ' '.join(seg_list)
    return token_list

class config:
    # Data
    DATA_PATH = '../data/all.csv'
    TEST_RATIO = 0.15
    CONTENT_FIELD = 'content'
    LABEL_FIELD = 'class_label'
    STOPWORDS_PATH = '../data/stopwords.txt'
    
    # Feature engineering
    ENABLE_TFIDF = True       # Use Tfidf instead of Count
    MIN_DF = 2                # Minimum word frequency
    MAX_FEAT = 5000           # Maximum features
    
    # Model
    MODEL = 'nb'          # {'nb':NaiveBayes, 'rf':RandomForest, 'lr':LogisticRegression ...}, check modeling.py for more details. 'nb' is highly recommanded!
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
    
    # Segmentation
    print('Processing dataset...')
    t1 = time.time()
    train_df[config.CONTENT_FIELD] = train_df[config.CONTENT_FIELD].apply(lambda x: clean(x))
    test_df[config.CONTENT_FIELD] = test_df[config.CONTENT_FIELD].apply(lambda x: clean(x))
    
    # Load stopwords
    stopwords_list = []
    f = open(config.STOPWORDS_PATH)
    lines = f.read()
    stopwords_list = lines.split('\n')
    f.close()

    # Feature engineering
    if config.ENABLE_TFIDF:
        vectorizer = TfidfVectorizer(min_df=config.MIN_DF, max_features=config.MAX_FEAT, stop_words=stopwords_list)
    else:
        vectorizer = CountVectorizer(min_df=config.MIN_DF, max_features=config.MAX_FEAT, stop_words=stopwords_list)
        
    train_dmt = vectorizer.fit_transform(train_df[config.CONTENT_FIELD])
    test_dmt = vectorizer.transform(test_df[config.CONTENT_FIELD])
    
    X_train = pd.DataFrame(data=train_dmt.toarray(), columns=vectorizer.get_feature_names() )
    X_test = pd.DataFrame(data=test_dmt.toarray(), columns=vectorizer.get_feature_names() )
    
    y_train = train_df[config.LABEL_FIELD].map(config.CLS2IDX).tolist()
    y_test = test_df[config.LABEL_FIELD].map(config.CLS2IDX).tolist()
    t2 = time.time()
    print('Processing finished, time consumption in ' + str(t2-t1) + 's.')


    
    # Modeling
    print('Start modeling...')
    t1 = time.time()
    if config.MODEL == 'nb':
        model = MultinomialNB()
        model.fit(X_train,y_train)
        y_score = model.predict(X_test)
    else:
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

