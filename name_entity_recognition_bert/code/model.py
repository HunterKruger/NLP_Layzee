import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Dropout, Bidirectional, LSTM
from tensorflow.keras import Model
from transformers import TFBertModel, BertConfig, optimization_tf 
from keras_crf import CRFModel
import config

def create_model(do_train=True, train_steps=None):

    # must load config file (.json) to allow BERT model return hidden states
    bert_config = BertConfig.from_pretrained(config.BASE_MODEL_PATH, output_hidden_states=True)
    bert_as_encoder = TFBertModel.from_pretrained(config.BASE_MODEL_PATH, config=bert_config, name='bert')  

    # input layers for BERT
    input_ids = Input(shape=(config.MAX_LEN,), name='input_ids', dtype='int32')
    attention_mask = Input(shape=(config.MAX_LEN,), name='attention_mask', dtype='int32')
    token_type_ids = Input(shape=(config.MAX_LEN,), name='token_type_ids', dtype='int32')

    # BERT as embedding 
    embedding = bert_as_encoder(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        token_type_ids=token_type_ids,
        return_dict=True,
        output_hidden_states=True
    )

    # downstream model
    dropout = Dropout(config.DROPOUT_RATE, name='dropout')(embedding.hidden_states[-1])
    bilstm = Bidirectional(LSTM(units=config.BILSTM_UNITS, return_sequences=True, recurrent_dropout=config.RECURRENT_DROPOUT_RATE), name='bilstm')(dropout)
    bilstm2 = Bidirectional(LSTM(units=config.BILSTM_UNITS, return_sequences=True, recurrent_dropout=config.RECURRENT_DROPOUT_RATE), name='bilstm2')(bilstm)
    dense = TimeDistributed(Dense(units=config.DENSE_UNITS, activation="relu"), name='t_dense')(bilstm2)  

    # init model
    base = Model(inputs=[input_ids, attention_mask, token_type_ids], outputs = [dense])

    # freeze BERT or not
    base.get_layer('bert').trainable = not config.FREEZE_WHOLE_BERT  # BERT is the 3th layer in the whole model

    # freeze specific BERT encoder layers
    to_freeze_list = config.FREEZE_BERT_LAYERS_LIST
    if len(to_freeze_list) >= 1:
        for i in to_freeze_list:
            base.get_layer('bert')._layers[0]._layers[1]._layers[0][i].trainable = False    

    # connect CRF
    model = CRFModel(base, len(config.UNIQUE_TAGS))
    model.build([tf.TensorShape([None, config.MAX_LEN]), tf.TensorShape([None, config.MAX_LEN]), tf.TensorShape([None, config.MAX_LEN])])

    if not do_train:
        return model

    optimizer, _ = optimization_tf.create_optimizer(
        init_lr=config.INIT_LR, 
        num_train_steps=train_steps,
        num_warmup_steps=config.NUM_WARMUP_STEPS,
        min_lr_ratio=config.MIN_LR_RATIO,
        weight_decay_rate=config.WEIGHT_DECAY_RATE,
        power=config.POWER
    )        
    model.compile(optimizer=optimizer, metrics=['acc'])

    return model

