import jieba
import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
import config
from sklearn.utils import shuffle as reset
from imblearn.under_sampling import RandomUnderSampler


def undersampling(df):
    rus = RandomUnderSampler(sampling_strategy='majority')
    X = df.drop([config.LABEL_FIELD], axis=1)
    y = df[config.LABEL_FIELD]
    X_res, y_res = rus.fit_resample(X, y)
    df_ = pd.concat([X_res, y_res], axis=1)
    df_.drop_duplicates(inplace=True)    
    df_ = df_.sample(frac=1)       
    df_ = df_.reset_index()
    df_.drop('index',axis=1,inplace=True)
    return df_

def train_test_split(data_df, test_size=0.2, shuffle=True, random_state=None):
    """
    Split dataframe into training set and test set.
    data_df->DataFrame: to be splited
    test_size->float: proportion for test set
    shuffle->bool: shuffle before split
    random_state->int: random seed 
    return: training set & test set
    """
    if shuffle:
        data_df = reset(data_df, random_state=random_state)

    train = data_df[int(len(data_df) * test_size):].reset_index(drop=True)
    test = data_df[:int(len(data_df) * test_size)].reset_index(drop=True)

    return train, test

def load_embedding(path=config.WORD2VEC_FILE):
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
    length = len(word2vec[0])
    rng_pad = np.random.RandomState(0)
    rng_oov = np.random.RandomState(1)
    vec_pad = (rng_pad.rand(length) - 0.5) * 2
    vec_oov = (rng_oov.rand(length) - 0.5) * 2
    word2vec.add_vector('<PAD>', vec_pad)
    word2vec.add_vector('<OOV>', vec_oov)
    return word2vec

class CustomDataset(tf.keras.utils.Sequence): 
    # customized dataset
    # implement ___len___ & __getitem__ function
                                  
    def __init__(self,
                 sents, 
                 sents2,
                 labels, 
                 word2vec,
                 batch_size=config.BATCH_SIZE,
                 max_len=config.MAX_LEN):
        
        self.sents = sents
        self.sents2 = sents2
        self.labels = labels
        self.batch_size = batch_size
        self.max_len = max_len
        self.word2vec = word2vec
        self.indexes = np.arange(len(self.sents))
    
    
    def __len__(self):  # return batch count
        return int(np.ceil(len(self.sents) / self.batch_size))
    
    
    def get_counts(self):
        return len(self.sents)

    
    def get_word2vec(self):
        return self.word2vec
    
    def segmentation(self, x):
        seg_list = jieba.cut(x)
        token_id_list = [self.word2vec.key_to_index[x] if x in self.word2vec.key_to_index.keys() else self.word2vec.key_to_index['<OOV>'] for x in seg_list] 
        return token_id_list
    
    
    def __getitem__(self, idx): # get tokenized sample by index
        
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        sents = self.sents[indexes]
        sents = list(map(lambda x: self.segmentation(x), sents))
        sents = tf.keras.preprocessing.sequence.pad_sequences(
            sents, 
            value=self.word2vec.key_to_index['<PAD>'], 
            padding='post',
            maxlen=self.max_len
        )

        sents2 = self.sents2[indexes]
        sents2 = list(map(lambda x: self.segmentation(x), sents2))
        sents2 = tf.keras.preprocessing.sequence.pad_sequences(
            sents2, 
            value=self.word2vec.key_to_index['<PAD>'], 
            padding='post',
            maxlen=self.max_len
        )
        
        if self.labels is None:
            return [sents, sents2]
        else:    
            labels = self.labels[indexes]
            return [sents, sents2], labels


# def process_text(sents, 
#                  fit_transform=True, 
#                  stopwords=config.STOPWORDS_FILE, 
#                  tokenizer_path=config.TOKENIZER_PATH,
#                  maxlen=config.MAX_LEN
#                 ):
#     '''
#     sents: a column of DataFrame, i.e. a Series
#     fit_transform: do fit and transform if True, else do transform
#     stopwords: remove stopwords if a path is specified
#     tokenizer_path: specify a path to save/load a tokenizer
#     '''
#     stopwords_list = []
#     if stopwords is not None:
#         f = open(config.STOPWORDS_FILE)
#         lines = f.read()
#         stopwords_list = lines.split('\n')
#         f.close()
    
#     def segmentation(x):
#         seg_list = jieba.cut(x)
#         token_list = [x for x in seg_list if x not in stopwords_list]
#         return token_list
    
#     sents = sents.apply(lambda x: segmentation(x))
#     sents = sents.tolist()
    
#     if fit_transform:
#         tokenizer=tf.keras.preprocessing.text.Tokenizer()
#         tokenizer.fit_on_texts(sents)                         # fit
#         ## save the tokenizer
#         tokenizer_json = tokenizer.to_json()
#         with io.open(tokenizer_path, 'w', encoding='utf-8') as f:
#             f.write(json.dumps(tokenizer_json, ensure_ascii=False))
#     else:
#         ## read the tokenizer
#         with open(tokenizer_path) as f:
#             data = json.load(f)
#             tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
            
#     sents = tokenizer.texts_to_sequences(sents)               # transform
#     sents = tf.keras.preprocessing.sequence.pad_sequences(sents, value=0, padding='post',maxlen=maxlen)

#     return sents, tokenizer


