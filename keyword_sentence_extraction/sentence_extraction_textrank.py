'''
Simple TextRank version. 
No need word2vec to calculate similarity. 
Keyword extraction and text summarization.
https://juejin.cn/post/6921978399368396807
'''

from textrank4zh import TextRank4Keyword
from textrank4zh import TextRank4Sentence


FILE_PATH = 'online_test.txt'
SENT_NB = 5


class TextRank:
    
    def __init__(self, input_path):
        
        with open(input_path, encoding='utf-8') as f:
            self.text = f.readlines()
        self.text = ''.join(self.text)

    def summary(self, nb=3):
        tr4s = TextRank4Sentence()
        tr4s.analyze(text=self.text, source='all_filters')
        print('Summary：')
        for item in tr4s.get_key_sentences(num=nb):
            print(item.index, item.weight, item.sentence)
            
def main():
    tt = TextRank(FILE_PATH)
    tt.summary(nb=SENT_NB)
    

if __name__ == "__main__":
    # execute only if run as a script
    main()
