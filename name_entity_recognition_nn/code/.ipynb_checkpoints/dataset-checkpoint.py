import re
import io
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from config import config
    
def process_text(data, 
                 maxlen=config.MAX_LEN, 
                 tokenizer_path=config.TOKENIZER_PATH,
                 with_labels=True, 
                 fit_transform=True):
    '''
    data: filepath, with filename
    maxlen: max length of sequence
    with_labels: data with labels or not
    fit_transform: fit then transform on the data if True, else only do transform
    tokenizer_path: path to save or load the tokenizer
    '''       
    if with_labels:
        # token_docs = [['a','b','c'],['a','b','c'],...]
        # tag_docs_in_digit = [[1, 2, 3], [1, 2, 3], ...]
        token_docs, tag_docs_in_digit = read_data_with_labels(data)
    else:
        token_docs = read_data_without_labels(data)
        fit_transform = False
        
    if fit_transform:
        tokenizer=tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(token_docs)                         # fit
        ## save the tokenizer
        tokenizer_json = tokenizer.to_json()
        with io.open(tokenizer_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    else:
        ## read the tokenizer
        with open(tokenizer_path) as f:
            data = json.load(f)
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
            
    token_docs = tokenizer.texts_to_sequences(token_docs)                # transform
    token_docs = tf.keras.preprocessing.sequence.pad_sequences(token_docs, value=0, padding='post', maxlen=maxlen)

    if with_labels == False:      
        return token_docs, tokenizer
    else:
        tag_docs_in_digit = tf.keras.preprocessing.sequence.pad_sequences(tag_docs_in_digit, value=config.TAG2ID['PAD'], padding='post', maxlen=maxlen)
        return token_docs, tag_docs_in_digit, tokenizer

            
def read_data_with_labels(data):
    # read data from 'data'
    file_path = Path(data)
    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    # print(raw_docs[1])
    for doc in raw_docs:
        tokens = []
        tags = []
        # print(doc)
        for line in doc.split('\n'):
            # print(line)
            token, tag = line.split(' ')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)
    tag_docs_in_id = [[config.TAG2ID[tag] for tag in tags] for tags in tag_docs]
    return token_docs, tag_docs_in_id


def read_data_without_labels(data):
    file_path = Path(data)
    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    # print(raw_docs[1])
    for doc in raw_docs:
        tokens = []
        for line in doc.split('\n'):
            token = line
            tokens.append(token)
        token_docs.append(tokens)
    return token_docs

