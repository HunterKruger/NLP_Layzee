import json
import pandas as pd
from config import config

if __name__ == "__main__":
    # execute only if run as a script

    with open(config.ORIGIN_FILE, 'r') as f:
        data = json.load(f)
        titles, contents = [], []
        for sample in data:
            titles.append(sample['title'])
            contents.append(sample['content'])

    df = pd.DataFrame()
    df['title']=titles
    df['content']=contents
    df['content_len'] = df['content'].apply(lambda x: len(x))
    df['title_len'] = df['title'].apply(lambda x: len(x))

    # MAX_LEN_DECODER - 1 : leave one place for end token
    # MAX_LEN_ENCODER - 2 : leave 2 place for start and end token
    df_filtered = df[(df['content_len']<=config.MAX_LEN_ENCODER-2) & (df['title_len']<=config.MAX_LEN_DECODER-1)].copy()  
    df_filtered.drop('title_len',axis=1,inplace=True)
    df_filtered.drop('content_len',axis=1,inplace=True)

    df_filtered.drop_duplicates(inplace=True)    
    df_filtered = df_filtered.sample(frac=1)       
    df_filtered = df_filtered.reset_index()
    df_filtered.drop('index',axis=1,inplace=True)

    print('Data shape:')
    print(df_filtered.shape)

    df_filtered.to_csv(config.ALL_FILE, index=False)

