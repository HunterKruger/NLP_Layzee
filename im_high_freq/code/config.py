import datetime

input_filename = 'input_data.txt'

cn_stopword_path     = "../dicts/stop_sents_customer_cn.txt"
en_stopword_path     = "../dicts/stop_sents_customer_en.txt"
common_stopword_path = "../dicts/cn_stopwords.txt"
keepword_path        = '../dicts/keep_words.txt'
synonyms_path        = '../dicts/synonyms.txt'
legal_pos            = ['n', 'nz', 'nt', 'v', 'vn', 'vd', 'l', 'a', 'd', 'eng', 'x']
tokenized_filename   = 'tokenized.txt'

tfidf_max_df       = 0.05  # 0.1
tfidf_max_features = 3000  # 2000
tfidf_top_k        = 7     # 5

bigrams_count          = 9999999
bigrams_occur          = 0.0003
bigrams_window         = 2
bigrams_show_samples   = 5
bigrams_time_limit     = 60*60  # 40
bigrams_result_df_path = 'bigram.txt'

trigrams_count          = 9999999
trigrams_occur          = 0.0001
trigrams_window         = 5
trigrams_show_samples   = 5
trigrams_time_limit     = 60*100  # 80
trigrams_result_df_path = 'trigram.txt'

standard_Q_path = '../dicts/StandardQ20220607.xlsx'
similar_Q_count = 3
pretrained_model_path = '../sbert/paraphrase-multilingual-MiniLM-L12-v2'

config_path = '../config/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.py'

tokenized_tablename = 'tmp_htl_ai_db.tmp_fy_im_weekly_tokenized'
bigram_tablename    = 'tmp_htl_ai_db.tmp_fy_im_weekly_bigrams'
trigram_tablename   = 'tmp_htl_ai_db.tmp_fy_im_weekly_trigrams'

adm_tokenized_tablename = 'htl_ai_db.adm_im_weekly_tokenized'
adm_bigram_tablename    = 'htl_ai_db.adm_im_weekly_bigrams'
adm_trigram_tablename   = 'htl_ai_db.adm_im_weekly_trigrams'