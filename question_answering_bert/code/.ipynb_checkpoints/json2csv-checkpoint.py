import json
import pandas as pd
from config import config

def ann_json2csv(input_path):

    with open(input_path, 'r') as f:
        data = json.load(f)

        questions, documents, answers, starts, ends = [], [], [], [], []

        for d in data:

            question = data[d]['question']
            document = data[d]['evidences'][d+'#00']['evidence']
            answer = data[d]['evidences'][d+'#00']['answer'][0]

            start = document.find(answer)
            end = start + len(answer)

            questions.append(question)
            documents.append(document)
            answers.append(answer)
            starts.append(start)
            ends.append(end)

        df = pd.DataFrame({'question':questions, 'document':documents, 'answer':answers, 'start':starts, 'end':ends})
        return df
    

def ir_json2csv(input_path, skip_no_answer = False):
    
    # 1 question
    # multiple documents
    # same answer
    # the answer may not be found in some documents!!

    with open(input_path, 'r') as f:
        
        data = json.load(f)

        questions, documents, answers, starts, ends = [], [], [], [], []

        for d in data:

            question = data[d]['question']
            evidences = data[d]['evidences']
            
            for evidence in evidences:
                
                answer = data[d]['evidences'][evidence]['answer'][0]
                document = data[d]['evidences'][evidence]['evidence']
                start = document.find(answer)
                end = start + len(answer)
                          
                if answer == 'no answer':
                    start = -1
                    end = -1

                questions.append(question)
                documents.append(document)
                answers.append(answer)
                starts.append(start)
                ends.append(end)

        df = pd.DataFrame({'question':questions, 'document':documents, 'answer':answers, 'start':starts, 'end':ends})
        if skip_no_answer:
            df = df[df['answer']!='no_answer']
            df = df[df['start']!=-1]

        return df
    
    
    
    
if __name__ == "__main__":
    # execute only if run as a script
    
    df1 = ann_json2csv('../origin_data/me_test.ann.json')
    df2 = ann_json2csv('../origin_data/me_validation.ann.json')

    df3 = ir_json2csv('../origin_data/me_test.ir.json', True)
    df4 = ir_json2csv('../origin_data/me_validation.ir.json', True)
    df5 = ir_json2csv('../origin_data/me_train.json', True)

    df = pd.concat([df1,df2,df3,df4,df5],axis=0)
    
    df = df[['question','document']]
    df.drop_duplicates(inplace=True)
    
    df = df.sample(frac=1)  # shuffle
    df = df.reset_index()
    df.drop('index',axis=1,inplace=True)

    print(df.shape)

    df['len_question'] = df['question'].apply(lambda x : len(x))
    df['len_document'] = df['document'].apply(lambda x : len(x))
    df['len'] = df['len_document'] + df['len_document']
    df = df[df['len']<=(config.MAX_LEN-3)]
    
    print(df.shape)
    # df_final.to_csv('../data/all.csv', index=False)


    
