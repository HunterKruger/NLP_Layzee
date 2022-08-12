import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from splitter_sampler import SplitterSampler
from modeling import Modeling
import fasttext


class config:
    
    # Data input
    DATA_PATH = '../data/all.csv'
    TEST_RATIO = 0.15
    CONTENT_FIELD = 'content'
    LABEL_FIELD = 'class_label'
    STOPWORDS_PATH = '../data/stopwords.txt'
    
    # Data output
    TRAIN_OUT = '../data/train_fasttext.txt'
    TEST_OUT = '../data/test_fasttext.txt'
    
    # Model output
    MODEL_PATH = "../model/fasttext_model.bin"
    
    # Hyperparameters
    LR = 0.1
    DIM = 100
    EPOCH = 5
    WORD_NGRAMS = 2
        
    # Label info
    LABELS = ['中立', '正向', '负向']
    CLASSES = len(LABELS)
    CLS2IDX, IDX2CLS = dict(), dict()
    for i, label in enumerate(LABELS):
        CLS2IDX[label]=i
        IDX2CLS[i]=label
    
    
def main():
    
    # Load data
    df = pd.read_csv(config.DATA_PATH)
    df.drop('id',axis=1,inplace=True)
    train_df, test_df = SplitterSampler.split_df(df, config.TEST_RATIO)
    train_df.reset_index(inplace=True,drop=True)
    test_df.reset_index(inplace=True,drop=True)
    
    # Load stopwords
    stopwords_list = []
    f = open(config.STOPWORDS_PATH)
    lines = f.read()
    stopwords_list = lines.split('\n')
    f.close()

    def clean(x):
        x = ''.join(re.findall('[\u4e00-\u9fa5]', x))                    # remove symbols, numbers and English
        seg_list = jieba.cut(x)                                          # segmentation
        token_list = [x for x in seg_list if x not in stopwords_list]
        token_list = ' '.join(token_list)
        return token_list
    
    # Segmentation and remove stopwords
    train_df[config.CONTENT_FIELD] = train_df[config.CONTENT_FIELD].apply(lambda x: clean(x))
    test_df[config.CONTENT_FIELD] = test_df[config.CONTENT_FIELD].apply(lambda x: clean(x))
        
    train_df[config.LABEL_FIELD] = train_df[config.LABEL_FIELD].map(config.CLS2IDX).tolist()
    test_df[config.LABEL_FIELD] = test_df[config.LABEL_FIELD].map(config.CLS2IDX).tolist()

    # save to fasttext supported version
    def format2fasttext(x):
        return '__label__'+str(x)
    
    train_df[config.LABEL_FIELD] = train_df[config.LABEL_FIELD].apply(lambda x: format2fasttext(x))
    test_df[config.LABEL_FIELD] = test_df[config.LABEL_FIELD].apply(lambda x: format2fasttext(x))
    
    with open(config.TRAIN_OUT, 'w') as f:
        for index, row in train_df.iterrows():
            f.write(row[config.LABEL_FIELD] + ' , ' + row[config.CONTENT_FIELD] + '\n')
            
    with open(config.TEST_OUT, 'w') as f:
        for index, row in test_df.iterrows():
            f.write(row[config.LABEL_FIELD] + ' , ' + row[config.CONTENT_FIELD] + '\n')
       
    # Train model
    model = fasttext.train_supervised(config.TRAIN_OUT, 
                                      lr=config.LR, 
                                      dim=config.DIM, 
                                      epoch=config.EPOCH, 
                                      word_ngrams=config.WORD_NGRAMS, 
                                      loss='softmax')
    model.save_model(config.MODEL_PATH)

    # Evaluation
    classifier = fasttext.load_model(config.MODEL_PATH)
    result = classifier.test(config.TEST_OUT)
    print("Accuracy:", result[1])
    
    
if __name__ == "__main__":
    # execute only if run as a script
    main()


