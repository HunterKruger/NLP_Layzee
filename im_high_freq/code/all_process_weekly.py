import re
import os
import ast
import json
import jieba
import emoji
import config
import random
import datetime
import numpy as np 
import pandas as pd
from time import time
from datetime import date
from zhconv import convert
from shutil import copyfile
import jieba.posseg as pseg
from functools import reduce 
from datetime import timedelta
from collections import Counter
from category_encoders.count import CountEncoder
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder


print('读取参数')
cn_stopword_path     = config.cn_stopword_path
en_stopword_path     = config.en_stopword_path
common_stopword_path = config.common_stopword_path
keepword_path        = config.keepword_path
synonyms_path        = config.synonyms_path
legal_pos            = config.legal_pos
tokenized_filename   = config.tokenized_filename

tfidf_max_df       = config.tfidf_max_df
tfidf_max_features = config.tfidf_max_features
tfidf_top_k        = config.tfidf_top_k

bigrams_count          = config.bigrams_count
bigrams_occur          = config.bigrams_occur
bigrams_window         = config.bigrams_window
bigrams_show_samples   = config.bigrams_show_samples
bigrams_time_limit     = config.bigrams_time_limit
bigrams_result_df_path = config.bigrams_result_df_path

trigrams_count          = config.trigrams_count
trigrams_occur          = config.trigrams_occur
trigrams_window         = config.trigrams_window
trigrams_show_samples   = config.trigrams_show_samples
trigrams_time_limit     = config.trigrams_time_limit
trigrams_result_df_path = config.trigrams_result_df_path

standard_Q_path       = config.standard_Q_path
pretrained_model_path = config.pretrained_model_path
similar_Q_count       = config.similar_Q_count
config_path           = config.config_path

tokenized_tablename = config.tokenized_tablename
bigram_tablename    = config.bigram_tablename
trigram_tablename   = config.trigram_tablename

adm_tokenized_tablename = config.adm_tokenized_tablename
adm_bigram_tablename    = config.adm_bigram_tablename
adm_trigram_tablename   = config.adm_trigram_tablename

input_filename = config.input_filename

t1=time()

df=pd.read_csv(input_filename, header=0,sep='\t', error_bad_lines=False, engine='python')
df = df[['session_id','datachange_createtime',  'msgtype', 'type', 'rank', 'body']]
print('原始数据集shape: ', df.shape)

print('1. 清洗和筛选')

print('对session_id进行计数编码')
df_session = df['session_id']
ce = CountEncoder(cols=['session_id'])
df_encoded = ce.fit_transform(df)
df_encoded.columns = ['round','datachange_createtime',  'msgtype', 'type', 'rank', 'body']
df = pd.concat([df_session, df_encoded], axis=1)

session_ids = df['session_id'].drop_duplicates()
session_ids_count = session_ids.shape[0]
print('总session数量', session_ids_count)

print('1.1 round完整性筛选')
rank1_session_ids = df[df['rank']==1]['session_id'].drop_duplicates()
print('有rank1的session数量：', rank1_session_ids.shape)
print('有rank1的session比例：', rank1_session_ids.shape[0] / session_ids.shape[0])   

rank_eq_round_session_ids = df[df['rank']==df['round']]['session_id'].drop_duplicates()
print('rank=round的session数量', rank_eq_round_session_ids.shape)
print('rank=round的session比例', rank_eq_round_session_ids.shape[0] / session_ids.shape[0] )       

session_ids = pd.merge(rank1_session_ids, session_ids, how='inner', on='session_id')
session_ids = pd.merge(rank_eq_round_session_ids, session_ids, how='inner', on='session_id')
print('完整的session数量：', session_ids.shape)
print('完整的session占比：', session_ids.shape[0] / session_ids_count)  # session 既完整也能进人工 比例

df = pd.merge(session_ids, df, how='left', on='session_id')

print('1.2 根据round内部发言筛选')
print('聚合type字段')
df_temp = df[['session_id', 'type']]
df_temp['type'] = df_temp['type'].apply(lambda x: str(x))
df_temp = df_temp.groupby(by='session_id')['type'].sum().to_frame()  # 聚合type
df_temp = df_temp.reset_index()
df_temp.columns = ['session_id', 'type_seq']

print('确保session中同时有顾客客服和机器人的发言')
def find_man_and_machine(x):  
    if ('1' in x and '2' in x and '3' in x) or ('1' in x and '2' in x and '4' in x):
        return True
    else:
        return False
df_temp['man_and_machine'] = df_temp['type_seq'].apply(lambda x: find_man_and_machine(x))
df_temp = df_temp[df_temp['man_and_machine']==True]
print('此时的session数量', df_temp.shape)


print('取最后的顾客发言之前的一个机器发言')
def find_last_machine(x):
    last1idx = len(x) - x[::-1].index('1') - 1
    x = x[:last1idx]
    if '2' in x:
        return len(x) - x[::-1].index('2') 
    else:
        return 0
    
df_temp['last_machine'] = df_temp['type_seq'].apply(lambda x: find_last_machine(x))
df = pd.merge(df_temp, df, how='left', on='session_id')
df = df[df['rank'] > df['last_machine']]
print('此时数据集shape: ', df.shape)

print('1.3 根据type和msgtype筛选')
type_dict = {1:'客人',2:'机器',3:"酒店客服人员",4:'携程客服人员', 5:'服务评价'}
msgtype_dict = {0:'手打文本',1:'图像',2:'酒店或房型',3:'视频',4:'语音',5:'文件或链接',6:'坐标',7:'json模板'} 
df = df[(df['type']==1) ]
df = df[(df['msgtype']==0) | (df['msgtype']==7)]
print('此时数据集shape: ', df.shape)

print('1.4 JSON 转文本')
def get_title(body):
    if 'title' in body and '{' in body and '}' in body:
        return json.loads(body, strict=False)['title']
    else:
        return body
df['body']=df['body'].apply(lambda x : get_title(x))

print('1.5 Session聚合')
df = df[['session_id','rank','body']]
df = df.groupby(by='session_id').body.sum().to_frame()  # 聚合body
df = df.reset_index()
print('此时数据集shape: ', df.shape)

print('删除转义符，防止保存csv的时候出错')
df['body'] = df['body'].apply(lambda x: x.replace('\r',''))  
df['body'] = df['body'].apply(lambda x: x.replace('\n',''))  
df['body'] = df['body'].apply(lambda x: x.replace('\t',''))  

print('2. 分词')
print('2.1 读取停止词')
f = open(cn_stopword_path,"r")   
stop_sents_cn = f.read()    
f.close()   
stop_sents_cn = stop_sents_cn.split('\n')
f = open(en_stopword_path,"r")   
stop_sents_en = f.read()    
f.close()   
stop_sents_en = stop_sents_en.split('\n')
f = open(common_stopword_path,"r")    
stop_sents_common = f.read()   
f.close()  
stop_sents_common = stop_sents_common.split('\n')
stop_words = list(set(stop_sents_cn+stop_sents_common+stop_sents_en))
print('停止词数量：',len(stop_words))

print('2.2 加载保留词')
jieba.load_userdict(keepword_path)   

print('2.2~ 加载同义词')
combine_dict = {}
for line in open(synonyms_path, "r"):
    seperate_word = line.strip().split(" ")
    num = len(seperate_word)
    for i in range(1, num):
        combine_dict[seperate_word[i]] = seperate_word[0]
print('同义词对数量：', len(combine_dict))

def clean_then_tokenize(x):
    x = str(x)
    x = emoji.replace_emoji(x, replace='')     # 去除emoji
    x = convert(x, 'zh-cn')                    # 繁简转换
    x = re.sub(u"\\(.*?\\)|\\{.*?\\}|\\[.*?\\]|\\<.*?\\>", "", x)  # 去除括号及其内部

    ## 分词，去停止词，同时pos筛选
    result = []
    words = pseg.cut(x)
    for word, flag in words:
        cn_word = re.findall('[\u4e00-\u9fa5]', word) # 只保留汉字
        cn_word = ''.join(cn_word) 
        if cn_word!='':
            word=cn_word
        word = word.lower()
        if (flag in legal_pos) and (word not in stop_words):
            result.append(word.rstrip().lstrip())
    return list(filter(lambda x : x != '', result))

def replace_synonyms(x):
    for key in combine_dict.keys():
        if key in x:
            x = x.replace(key, combine_dict[key])
    return x

print('2.2 开始分词')
df['body'] = df['body'].apply(lambda x: replace_synonyms(x)) 
df['tokens_list'] = df['body'].apply(lambda x: clean_then_tokenize(x))
df['tokens'] = df['tokens_list'].apply(lambda x: ' '.join(x))
df = df.drop(df[df['tokens']==''].index)

df.reset_index(inplace=True)
df.drop('index',axis=1,inplace=True)
print('当前数据量', df.shape)

print('保存分词后的数据')
df.insert(0, 'org_date_start', datetime.date.today()+timedelta(days=-8))
df.insert(0, 'org_date_end', datetime.date.today()+timedelta(days=-2))
df.to_csv(tokenized_filename, header=False, encoding='utf-8-sig', index=False, sep='\t')

print('3. TF-IDF建模：')
df['idx'] = df.index
tfidf_model = TfidfVectorizer(
    stop_words=stop_words,
    token_pattern=r"(?u)\b\w+\b",  # enable 1-char words
    lowercase=False,
    max_df=tfidf_max_df, 
    max_features=tfidf_max_features  # only depends on tf
)

weight = tfidf_model.fit_transform(df['tokens']).toarray()
word = tfidf_model.get_feature_names()
df_word_weight = pd.DataFrame(data=weight, columns=word)
topwords = df_word_weight.sum(axis=0)
print('TFIDF后的数据大小', df_word_weight.shape)
print('Topwords:')
print(topwords.index.tolist())

print('3.1 提取每个session的关键词：')
def get_keywords(x, topK=tfidf_top_k):
    key_words = []
    for i, w in enumerate(weight[x]):
        if w>0 :
            key_words.append( (word[i],float('{:.3f}'.format(w))) )

    key_words.sort(key=lambda y: y[1], reverse=True)     
    result = key_words[:topK]
    return [x for x,_ in result]

df['keywords'] = df['idx'].apply(lambda x: get_keywords(x))
df.drop('idx', inplace=True, axis=1)

t2=time()
print('预处理耗时：', t2-t1)


def find_stardard_Q(keys, Qs):
    keys = keys.split('+')
    for Q in Qs:
        if all(key in Q for key in keys):
            return Q
    return np.nan


def add_stardard_Q(df, standard_Q_path):
    standard_Q = pd.read_excel(standard_Q_path,engine='openpyxl')
    Qs = standard_Q['业务Q'].tolist()
    Qs = list(set(Qs))
    print('标准Q数量：', len(Qs))
    df['standard_Q'] = df['gram'].apply(lambda x: find_stardard_Q(x, Qs))
    print('新挖掘问题数量',len(df))
    print('不在标准Q里的数量',df['standard_Q'].isna().sum())
    print('占比',df['standard_Q'].isna().sum()/len(df))
    return df

    
    
def get_bigrams_report(df, topwords, result_df_path, standard_Q_path, time_limit=600, show_samples=10, grams_count=100, occur=0.0001, window=2):
    
    t1=time()
    
    # input for calculating bigram pmi scores
    data_pmi_input = []
    for words in df['tokens_list'].tolist():
        data_pmi_input.extend(words)
        for i in range(window-1):
            data_pmi_input.extend('|')
    
    # calculate bigram pmi scores
    bigram_measures = BigramAssocMeasures()
    bi_finder = BigramCollocationFinder.from_words(data_pmi_input,window_size=window)  # default window size = 2 
    
    # save bigram pmi scores and generate report
    bigrams = []
    result_df = pd.DataFrame()
    if 0<occur<1:
        occur = len(df)*occur
 
    for w1w2, score in bi_finder.score_ngrams(bigram_measures.pmi): # iterate grams and its score
        
        t2=time()
        if t2-t1>time_limit or grams_count==0:
            break
            
        w1, w2 = w1w2
        list_w1w2 = sorted(set([w1, w2]))
        
        if  (w1!=w2) and (w1 in topwords) and (w2 in topwords) and len(list_w1w2)==2 and (list_w1w2 not in bigrams):
            bigrams.append(list_w1w2)
            temp_df = df['keywords'].apply(set(list_w1w2).issubset)
            cnt = temp_df.sum()
            row_result = dict()
            
            if cnt>occur:
                print(list_w1w2,grams_count)
                row_result['gram']=w1+'+'+w2
                row_result['pmi']=score
                row_result['occurance']=cnt
                sample_df = df.iloc[temp_df[temp_df==True].index][:show_samples]
                sample_df = sample_df[['session_id','body']]
                row_result['samples'] = dict(zip(sample_df.session_id, sample_df.body))
                result_df = result_df.append(row_result, ignore_index=True)
                grams_count-=1
        
    
    result_df = add_stardard_Q(result_df, standard_Q_path)
    result_df = result_df[['gram', 'occurance', 'pmi', 'standard_Q', 'samples']]
    result_df.insert(0, 'org_date_start', datetime.date.today()+timedelta(days=-8))
    result_df.insert(0, 'org_date_end', datetime.date.today()+timedelta(days=-2))
    result_df.to_csv(result_df_path, header=True, index=False, encoding='utf-8-sig', sep='\t')  # header=True, 记得写数据的时候要 False
    print(len(result_df), ' grams')
    return result_df

print('4.1 提取Bigrams：')

result_bigram = get_bigrams_report(
    df, 
    topwords, 
    grams_count=bigrams_count, 
    occur=bigrams_occur, 
    window=bigrams_window,
    show_samples=bigrams_show_samples, 
    time_limit=bigrams_time_limit,
    result_df_path=bigrams_result_df_path, 
    standard_Q_path=standard_Q_path
)

def get_trigrams_report(df, bigrams, result_df_path, standard_Q_path, time_limit=1000, grams_count=200, show_samples=5, occur=0.0001, window=5):

    t1=time()
    
    # input for calculating trigram pmi scores
    data_pmi_input = []
    for words in df['tokens_list'].tolist():
        data_pmi_input.extend(words)
        for i in range(window-1):
            data_pmi_input.extend('|')
    
    # calculate trigram pmi scores
    trigram_measures = TrigramAssocMeasures()
    tri_finder = TrigramCollocationFinder.from_words(data_pmi_input,window_size=window) 
    
    # save trigram pmi scores # generate report
    trigrams = []
    result_df = pd.DataFrame()
    if 0<occur<1:
        occur = occur*len(df)
    for w1w2w3, score in tri_finder.score_ngrams(trigram_measures.pmi):
        t2=time()
        if grams_count==0 or t2-t1>time_limit:
            break
        is_trigram_recorded = False
        w1, w2, w3 = w1w2w3
        set_w1w2w3 = set([w1,w2,w3])
        list_w1w2w3 = sorted(set_w1w2w3)
        
        for set_w1w2 in bigrams:
            if set_w1w2.issubset(set_w1w2w3) and len(set_w1w2w3)==3 and (list_w1w2w3 not in trigrams):
                trigrams.append(list_w1w2w3)
                is_trigram_recorded=True
                break
            
        if is_trigram_recorded:
            temp_df = df['keywords'].apply(set_w1w2w3.issubset)
            cnt = temp_df.sum()
            row_result = dict()
            if cnt>occur:
                print(list_w1w2w3, grams_count)
                row_result['gram']=w1+'+'+w2+'+'+w3
                row_result['pmi']=score
                row_result['occurance']=cnt
                sample_df = df.iloc[temp_df[temp_df==True].index][:show_samples]
                sample_df = sample_df[['session_id','body']]
                row_result['samples'] = dict(zip(sample_df.session_id, sample_df.body))
                result_df = result_df.append(row_result, ignore_index=True)
                grams_count-=1

    result_df = add_stardard_Q(result_df, standard_Q_path)
    result_df = result_df[['gram', 'occurance', 'pmi', 'standard_Q', 'samples']]
    result_df.insert(0, 'org_date_start', datetime.date.today()+timedelta(days=-8))
    result_df.insert(0, 'org_date_end', datetime.date.today()+timedelta(days=-2))

    result_df.to_csv(result_df_path, header=True, index=False, encoding='utf-8-sig', sep='\t')  # header=True, 记得写数据的时候要 False
    print(len(result_df), ' grams')
    return result_df

bigrams = result_bigram.gram.tolist()
bigrams = [set(x.split('+')) for x in bigrams]

print('4.2 提取Trigrams：')
result_trigram=get_trigrams_report(
    df, bigrams, 
    result_df_path=trigrams_result_df_path,
    standard_Q_path=standard_Q_path, 
    time_limit=trigrams_time_limit,
    grams_count=trigrams_count, 
    show_samples=trigrams_show_samples, 
    occur=trigrams_occur, 
    window=trigrams_window
)

def add_similar_Q(result_df_path, standard_Q_path, pretrained_model_path):
    df = pd.read_csv(result_df_path, sep='\t', encoding='utf-8-sig')
    df.columns = ['org_date_end','org_date_start','gram','occurance','pmi','standard_Q','samples']
    standard_Q = pd.read_excel(standard_Q_path, engine='openpyxl')
    model = SentenceTransformer(pretrained_model_path)
    Qs = standard_Q['业务Q'].tolist()
    Qs = list(set(Qs))
    passage_embedding = model.encode(Qs)
    
    def find_similar_Qs(x):
        query_embedding = model.encode(x)
        results = util.dot_score(query_embedding, passage_embedding)[0].numpy()
        results = list(map(lambda x, y:(x,y), Qs, results))
        results = sorted(results, key=lambda x: x[-1], reverse=True)
        results = results[:similar_Q_count]
        qs = ''
        for q, score in results:
            qs+=q
            qs+='('
            qs+=str(round(score,2))
            qs+=') '
        return qs
    
    df['similar_Q'] = df['gram'].apply(lambda x: find_similar_Qs(x)) 
    df['standard_Q'].fillna('None',inplace=True) 
    
    def merge_Q(standard, similar):
        if standard=='None':
            return '相似匹配：' + similar
        else:
            return '精准匹配：' + standard
    
    df['standard_Q'] = df.apply(lambda x: merge_Q(x['standard_Q'], x['similar_Q']), axis=1)
    df.drop('similar_Q', axis=1, inplace=True)
    df.sort_values('occurance', ascending=False, inplace=True)
    df.to_csv(result_df_path, sep='\t', encoding='utf-8-sig', index=False, header=True)


### 预训练模型模糊匹配标准Q ###
print('4.3 预训练模型模糊匹配标准Q：')
add_similar_Q(bigrams_result_df_path , standard_Q_path, pretrained_model_path)
add_similar_Q(trigrams_result_df_path, standard_Q_path, pretrained_model_path)
### 预训练模型模糊匹配标准Q ###


### 加规则合并类似的词组 ###
print('4.4 加规则合并类似词组：')
tri_df = pd.read_csv(trigrams_result_df_path,encoding='utf-8-sig', sep='\t')
tri_df['samples'] = tri_df['samples'].apply(lambda x: str(x))
# 人工匹配标准Q
merge_newQ_dict = {
    '房间里可以投屏嘛？':['电视+手机+投屏','房间+电视+投屏','电视+支持+手机','电视+支持+投屏'],
    '可以开发票吗？':[
        '开+电子+发票','名称+有限公司+税号','电子+发票+qq','有限公司+税号+邮箱','开户行+账号+支行','帮忙+开+发票','开发票+开+发票', '开发票+专票+开',
        '发票+邮箱+qq', '有限公司+邮箱+qq',                                                                    # 20220705
        '电子+发票+邮箱','有限公司+税号+qq','开+发票+邮箱','开发票+电子+发票' ,'能开+专票+开','开+发票+qq'         # 20220719
    ],
    '疫情期间入住政策': [
        '行程码+带星+低风险', '疫情+入住+政策','小时+核酸+做', '需要+小时+核酸报告', '低风险+地区+隔离', 
        '做+核酸+没', '接待+居家+隔离', '行程码+小时+核酸', '入住+防疫+要求', '提供+小时+核酸',
        '入住+核酸+检测证明','低风险+地区+核酸', '低风险+地区+带星', '小时+核酸+绿码', '低风险+地区+接待',
        '低风险+小时+核酸','低风险+地区+行程码', '低风险+地区+行程', '街道+社区+报备', '低风险+地区+小时',
        '低风险+地区+政策', '带星+小时+核酸', '低风险+中风险+地区', '低风险+地区+区', '低风险+核酸+阴性',
        '做+核酸+出', '绿码+行程码+带星', '行程码+带星+高风险', '低风险+地区+住', '区+中风险+地区',
        '低风险+行程+区', '行程码+带星+接待', '地区+小时+核酸',
        '低风险+地区+核酸报告','行程码+带星+隔离','地区+核酸码+带星','行程码+带星+核酸','绿码+低风险+区',                              # 20220705
        '小时+核酸+做核酸','要求+小时+核酸','行程码+带星+做','绿码+带星+核酸','接待+带星+低风险',                                     # 20220705 
        '政策+行程码+带星', '健康码+行程码+带星', '核酸报告+做+核酸', '做+核酸+没出', '低风险+区+带星', '做+完+核酸',                  # 20220705
        '低风险+地区+核酸检测', '小时+核酸+阴性', '行程码+带星+没', '低风险+接待+行程码', '绿码+带星+低风险', '健康码+绿码+行程码',     # 20220705
        '小时+核酸报告+核酸', '核酸+行程码+绿码', '带星+高风险+地区', '低风险+社区+报备', '疫情+政策+低风险', '低风险+绿码+核酸',       # 20220705
        '接待+低风险+客人', '疫情+行程码+带星', '低风险+地区+疫情', '提供+小时+核酸报告', '行程码+带星+核酸报告', '低风险+地区+要求',   # 20220705
        '低风险+接待+隔离', '绿码+核酸+做', '低风险+区+隔离', '行程码+带星+报备', '要求+小时+核酸',                                   # 20220705
        '低风险+地区+街道', '高风险+地区+低风险', '街道+小区+低风险', '低风险+街道+中风险', '低风险+区+政策', '高风险+地区+核酸',               # 20220715   
        '街道+中风险+小区', '疫情+政策+查看', '低风险+区+小时', '提示+疫情+政策', '低风险+地区+报备', '高风险+地区+街道',                       # 20220715   
        '高风险+地区+隔离', '防疫+政策+低风险', '高风险+区+低风险', '政策+高风险+地区', '街道+低风险+区', '接待+高风险+地区',                   # 20220715   
        '高风险+区+街道', '防疫+政策+地区', '防疫+政策+核酸', '中风险+小区+低风险', '高风险+地区+区', '低风险+地区+绿码', '低风险+街道+接待',     # 20220715   
        '做核酸+做+核酸', '低风险+地区+街道', '高风险+地区+低风险', '街道+小区+低风险', '低风险+街道+中风险', '低风险+区+政策',     # 20220719
        '高风险+地区+核酸', '街道+中风险+小区', '疫情+政策+查看', '低风险+区+小时','提示+疫情+政策','低风险+地区+报备',             # 20220719
        '高风险+地区+街道', '高风险+地区+隔离', '防疫+政策+低风险','小时+核酸+落地','高风险+区+低风险','政策+高风险+地区',          # 20220719
        '街道+低风险+区', '接待+高风险+地区','高风险+区+街道','防疫+政策+地区','防疫+政策+核酸', '高风险+街道+低风险',              # 20220719
        '街道+中风险+地区','中风险+小区+低风险','高风险+地区+区','疫情+防控+要求','低风险+地区+绿码', '防疫+政策+要求',             # 20220719
        '低风险+街道+接待','高风险+地区+区域','街道+高风险+小区','接待+低风险+区','低风险+地区+防疫','小时+核酸报告+做'             # 20220719
    ]   
}
new_merge_newQ_dict = dict()
for key,values in merge_newQ_dict.items():
    tmp_grams = []
    for grams in values:
        gram_list = grams.split('+')
        tmp_grams.append(gram_list[0]+'+'+gram_list[1]+'+'+gram_list[2])
        tmp_grams.append(gram_list[0]+'+'+gram_list[2]+'+'+gram_list[1])
        tmp_grams.append(gram_list[1]+'+'+gram_list[0]+'+'+gram_list[2])
        tmp_grams.append(gram_list[1]+'+'+gram_list[2]+'+'+gram_list[0])
        tmp_grams.append(gram_list[2]+'+'+gram_list[1]+'+'+gram_list[0])
        tmp_grams.append(gram_list[2]+'+'+gram_list[0]+'+'+gram_list[1])
    new_merge_newQ_dict[key] = list(set(tmp_grams))
print('人工匹配的字典：')
print(new_merge_newQ_dict)
# 添加人工匹配的Q
for key,values in new_merge_newQ_dict.items():
    for value in values:
        if tri_df[tri_df.gram==value].shape[0]==1:   # 存在这个三元组
            tri_df.loc[tri_df.gram==value, 'standard_Q'] = '人工匹配: ' + key 
# 合并相同人工匹配Q的三元组
for key,values in new_merge_newQ_dict.items():    
    tmp_df = tri_df[tri_df['standard_Q']=='人工匹配: '+ key]
    if len(tmp_df)>1:
        print(tmp_df)
        tmp_df_merged = dict()
        tmp_df_merged['org_date_start']=tmp_df['org_date_start'].tolist()[0]
        tmp_df_merged['org_date_end']=tmp_df['org_date_end'].tolist()[0]
        tmp_df_merged['standard_Q']=tmp_df['standard_Q'].tolist()[0]
        tmp_df_merged['occurance']=tmp_df['occurance'].sum()
        tmp_df_merged['pmi']=999999
        #tmp_df_merged['gram']='|'.join(tmp_df['gram'].tolist())
        #tmp_df_merged['samples']='|'.join(tmp_df['samples'].tolist())
        tmp_df_merged['gram']=tmp_df['gram'].tolist()[0] + ' ...'
        tmp_df_merged['samples']=tmp_df['samples'].tolist()[0] + ' ...'
        tri_df = tri_df.drop(index=tmp_df.index)
        tri_df = tri_df.append(tmp_df_merged, ignore_index=True)
tri_df.sort_values('occurance', ascending=False, inplace=True)
tri_df.to_csv(trigrams_result_df_path, header=False, index=False, encoding='utf-8-sig', sep='\t')   ## 记得部署时候改header=False
### 加规则合并类似的词组 ###


print('全部处理完毕！')

print('session数量', df.shape)
print('bigram数量' , result_bigram.shape)
print('trigram数量', result_trigram.shape)

print('保存配置文件：')
copyfile('config.py', config_path) 

t1 = time()
print('数据存入临时表：')
os.system(
 """hive -e "
 LOAD DATA LOCAL INPATH '{}' OVERWRITE INTO TABLE {};" """.format(tokenized_filename, tokenized_tablename)
)
os.system(
 """hive -e " 
 LOAD DATA LOCAL INPATH '{}' OVERWRITE INTO TABLE {};" """.format(bigrams_result_df_path, bigram_tablename)
)
os.system(
 """hive -e "
 LOAD DATA LOCAL INPATH '{}' OVERWRITE INTO TABLE {};" """.format(trigrams_result_df_path, trigram_tablename)
)


today = date.today()
d = today.strftime("%Y-%m-%d")

print('临时表数据写入正式表：')
os.system(
 """hive -e "
    insert overwrite table {} partition(d='{}')
    select * from {};
 " """.format(adm_bigram_tablename, d, bigram_tablename)
)

os.system(
 """hive -e "
    insert overwrite table {} partition(d='{}')
    select * from {};
 " """.format(adm_trigram_tablename, d, trigram_tablename)
)
os.system(
 """hive -e "
    insert overwrite table {} partition(d='{}')
    select * from {};
 " """.format(adm_tokenized_tablename, d, tokenized_tablename)
)


t2 = time()
print('结果写入数据库耗时', t2-t1)

print('脚本执行完毕！')

