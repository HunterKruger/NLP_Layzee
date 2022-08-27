import config

from transformers import TFBertModel, BertConfig
from tensorflow.keras import Input
from tensorflow.keras.layers import GlobalMaxPooling1D, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model

    
def create_model(
    nb_classes,
    dropout_rate=config.DROPOUT_RATE,
    freeze_bert_layers_list=config.FREEZE_BERT_LAYERS_LIST,
    freeze_whole_bert=config.FREEZE_WHOLE_BERT,
    hidden_state_list=config.HIDDEN_STATE_LIST
):    
      
    # must load config file (.json) to allow BERT model return hidden states
    bert_config = BertConfig.from_pretrained(config.BASE_MODEL_PATH, output_hidden_states=True,return_dict=True)
    bert_as_encoder = TFBertModel.from_pretrained(config.BASE_MODEL_PATH, config=bert_config, name='bert')  

    # input layers for BERT
    input_ids = Input(shape=(config.MAX_LEN,), name='input_ids', dtype='int32')
    attention_mask = Input(shape=(config.MAX_LEN,), name='attention_mask', dtype='int32')
    token_type_ids = Input(shape=(config.MAX_LEN,), name='token_type_ids', dtype='int32')

    # bert_model
    embedding = bert_as_encoder(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)

    ### downstream model
    if len(hidden_state_list) > 1:
        maxpool_list = []
        for i in range(len(hidden_state_list)):
            maxpool = GlobalMaxPooling1D(name='maxpool_'+str(i))(embedding.hidden_states[i])
            maxpool_list.append(maxpool)
        concat = Concatenate(name='concat')(maxpool_list)             # get the 0st token's hidden state then concatenate
        dropout = Dropout(dropout_rate, name='dropout')(concat)
    else:
        maxpool = GlobalMaxPooling1D(name='maxpool')(embedding.hidden_states[hidden_state_list[0]])
        dropout = Dropout(dropout_rate, name='dropout')(maxpool)

    output = Dense(nb_classes, activation='softmax', name='output')(dropout)  # output layer for multi-clf
    model = Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=output)

    # freeze BERT or not
    model.get_layer('bert').trainable = not freeze_whole_bert  

    # freeze specific BERT encoder layers
    if len(freeze_bert_layers_list) >= 1:
        for i in freeze_bert_layers_list:
            model.get_layer('bert')._layers[0]._layers[1]._layers[0][i].trainable = False     

    return model
