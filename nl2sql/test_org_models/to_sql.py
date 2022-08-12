from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelWithLMHead

class Text2SQL:

    def __init__(self, t5_path):
        self.tokenizer_t5 = AutoTokenizer.from_pretrained(t5_path)
        self.model_t5 = AutoModelWithLMHead.from_pretrained(t5_path)


    def text2sql(self, query):
        input_text = "translate English to SQL: %s </s>" % query
        features = self.tokenizer_t5([input_text], return_tensors='pt')
        output = self.model_t5.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'])
        result = self.tokenizer_t5.decode(output[0])
        result = result.replace('<pad>','')
        result = result.strip()
        return result

    def do_inference(self, input_sent, table_id):
        result = self.text2sql(input_sent)
        result = result.replace('</s>','')
        result = result.replace('table', table_id)
        return result

# example
t5_path =  "./t5-base-finetuned-wikiSQL"
text2sql = Text2SQL(t5_path)

input_sent = 'How many people in Shanghai？'
table_id = '123124'
print(text2sql.do_inference(input_sent, table_id))

input_sent = 'What is the biggest city in the United States?'
table_id = 'fsadf'
print(text2sql.do_inference(input_sent, table_id))