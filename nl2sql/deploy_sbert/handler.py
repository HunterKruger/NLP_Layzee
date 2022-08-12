
# https://medium.com/analytics-vidhya/deploy-huggingface-s-bert-to-production-with-pytorch-serve-27b068026d18

from abc import ABC
import json
import logging
import os

import torch
from ts.torch_handler.base_handler import BaseHandler
from sentence_transformers import SentenceTransformer, util

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = ''

logger = logging.getLogger(__name__)


class Sbert(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(Sbert, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        logger.info('model_dir:')
        logger.info(model_dir) 

        # model_dir = '/home/fengyuan/workspaceGServer/NLP/SPD_BANK/deploy_sbert/distiluse-base-multilingual-cased'

        # self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = SentenceTransformer(model_dir)


        # self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')

        self.initialized = True

    def preprocess(self, data):
        """ Very basic preprocessing code - only tokenizes. 
            Extend with your own preprocessing steps as needed.
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentences = text.decode('utf-8')
        logger.info("Received text: '%s'", sentences)

        inputs = self.tokenizer.encode_plus(
            sentences,
            add_special_tokens=True,
            return_tensors="pt"
        )
        return inputs


    def find_similar(self, query, corpus):
        '''
        query: str
        corpus: list of str
        return
            target sentence, socre
        '''
        corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True)
        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        top_k = min(1, len(corpus))

        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        return corpus[top_results[1]], round(top_results[0][0].item(),4)



    def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """
        # NOTE: This makes the assumption that your model expects text to be tokenized  
        # with "input_ids" and "token_type_ids" - which is true for some popular transformer models, e.g. bert.
        # If your transformer model expects different tokenization, adapt this code to suit 
        # its expected input format.
        
        logger.info('inputs_fy:')
        logger.info(inputs) # ?
        logger.info(type(inputs)) # ?

        logger.info(inputs[0].get('body'))  # ?
        logger.info(type(inputs[0].get('body')))  # ?

        # logger.info(inputs[0].get('body').decode('utf-8'))  # ?

    
        ## todo get query, corpus from inputs
        json_ = inputs[0].get('body')  # json
        query = json_['q']
        sep = json_.get('sep', '!~^')
        corpus = json_.get('list_str').split(sep)
        
        
        target_sent, score = self.find_similar(query, corpus)

        # prediction = self.zh2en(inputs[0].get('body').decode('utf-8'))
        # prediction = self.model(
        #     inputs['input_ids'].to(self.device), 
        #     token_type_ids=inputs['token_type_ids'].to(self.device)
        # )[0].argmax().item()
        # logger.info("Model predicted: '%s'", prediction)

        # if self.mapping:
        #     prediction = self.mapping[str(prediction)]
        prediction = dict()
        target_score = dict()
        target_score[target_sent]=score
        prediction[query]=target_score

        return [prediction]

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output


_service = Sbert()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

    #    data = _service.preprocess(data)
        data = _service.inference(data)
    #    data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
