from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelWithLMHead

class Text2SQL:

    def __init__(self, t5_path, mt_path):
        self.tokenizer_t5 = AutoTokenizer.from_pretrained(t5_path)
        self.model_t5 = AutoModelWithLMHead.from_pretrained(t5_path)
        self.tokenizer_mt = AutoTokenizer.from_pretrained(mt_path)
        self.model_mt = AutoModelForSeq2SeqLM.from_pretrained(mt_path)

    def zh2en(self, sentence):
        self.tokenizer_mt.padding_side = "left"
        self.tokenizer_mt.pad_token = self.tokenizer_mt.eos_token # to avoid an error
        task_prefix = 'translate Chinese to English: '
        inputs = self.tokenizer_mt(task_prefix + sentence, return_tensors="pt", padding=True)

        output_sequences = self.model_mt.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            do_sample=False, # disable sampling to test if batching affects output
        )
   

        result = self.tokenizer_mt.batch_decode(output_sequences, skip_special_tokens=True)[0]
        result = result.replace('<pad>','')
        result = result.strip()
        return result

    def text2sql(self, query):
        input_text = "translate English to SQL: %s </s>" % query
        features = self.tokenizer_t5([input_text], return_tensors='pt')
        output = self.model_t5.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'])
        result = self.tokenizer_t5.decode(output[0])
        result = result.replace('<pad>','')
        result = result.strip()
        return result

    def do_inference(self, input_sent, table_id):
        result = self.zh2en(input_sent)
        result = self.text2sql(result)
        result = result.replace('</s>','')
        result = result.replace('table', table_id)
        return result

# example
t5_path =  "../experiments/model/t5-base-finetuned-wikiSQL"
mt_path =  "../experiments/model/opus-mt-zh-en"
text2sql = Text2SQL(t5_path,mt_path)

query = '上海有多少人口？'
table_id = '123124'
print(text2sql.do_inference(query, table_id))

query = '美国最大的城市是哪个？'
table_id = 'fsadf'
print(text2sql.do_inference(query, table_id))