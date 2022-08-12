'''
Simple TextRank version. 
No need word2vec to calculate similarity. 
Keyword extraction and text summarization.
https://juejin.cn/post/6921978399368396807
'''

from textrank4zh import TextRank4Keyword
from textrank4zh import TextRank4Sentence


FILE_PATH = 'online_test.txt'
WORD_NB = 10
WINDOW = 3
WORD_MIN_LEN = 2

class TextRank:
    
    def __init__(self, input_path):
        
        with open(input_path, encoding='utf-8') as f:
            self.text = f.readlines()
        self.text = ''.join(self.text)
        
    def keyword(self, nb=5, window=3, word_min_len=2):
        tr4w = TextRank4Keyword()
        tr4w.analyze(text=self.text, window=window)
        print('Keywords：')
        for item in tr4w.get_keywords(nb, word_min_len=word_min_len):
            print(item.word, item.weight)
            
def main():
    tt = TextRank(FILE_PATH)
    tt.keyword(nb=WORD_NB, window=WINDOW, word_min_len=WORD_MIN_LEN)
    

if __name__ == "__main__":
    # execute only if run as a script
    main()
