from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class MT:

    def __init__(self, mt_path):
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


# example
mt_path =  "../experiments/model/opus-mt-zh-en"
mt = MT(mt_path)

query = '上海有多少人口？'
print(mt.zh2en(query))

query = '美国最大的城市是哪个？'
print(mt.zh2en(query))
