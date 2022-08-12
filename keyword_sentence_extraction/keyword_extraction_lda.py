from gensim import corpora, models
import jieba.posseg as jp
import jieba

FILE_PATH = 'online_test.txt'
NB_TOPICS = 3
PASSES = 10
TOP_WORD_PER_TOPIC = 10
NB_KEYWORD = 5
FLAGS = ['n', 'nr', 'ns', 'nt', 'eng', 'v', 'd']
STOPWORDS = ['的', '就', '是', '用', '还', '在', '上', '作为']


def get_text(text):
    flags = set(FLAGS)  
    stopwords = set(STOPWORDS)  
    words_list = []
    for text in texts:
        words = [w.word for w in jp.cut(text) if w.flag in flags and w.word not in stopwords]
        words_list.append(words)
    return words_list
 
    
def LDA_model(words_list):
    dictionary = corpora.Dictionary(words_list)
    corpus = [dictionary.doc2bow(words) for words in words_list]
    lda_model = models.ldamodel.LdaModel(corpus=corpus, num_topics=NB_TOPICS, id2word=dictionary, passes=PASSES)
 
    return lda_model
 
    
if __name__ == "__main__":
    

    with open(FILE_PATH,'r') as f:
        texts = f.read()
    texts = texts.split('。')

    words_list = get_text(texts)
    lda_model = LDA_model(words_list)
    topic_words = lda_model.print_topics(num_topics=NB_TOPICS)
    word2score = dict()

    for i in range(NB_TOPICS):
        words_list = lda_model.show_topic(i, TOP_WORD_PER_TOPIC)
        for word, score in words_list:
            if word in word2score.keys():
                word2score[word]+=score
            else:
                word2score[word]=score

    word2score = sorted(word2score.items(), key=lambda x: x[1], reverse=True)
    print(word2score[:NB_KEYWORD])