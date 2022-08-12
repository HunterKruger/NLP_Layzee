import jieba.analyse


FILE_PATH = 'online_test.txt'
WORD_NB = 10


with open(FILE_PATH,'r') as f:
    sentence = f.read()

print(jieba.analyse.extract_tags(sentence, topK=WORD_NB, withWeight=True, allowPOS=()))
