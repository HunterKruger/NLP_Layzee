import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Dropout, Bidirectional, LSTM, Concatenate
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from transformers import TFBertModel, BertConfig, optimization_tf 
from keras_crf import CRFModel

from config import config

def create_model(do_train=True, train_steps=None, summary=True):
    
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        
        # must load config file (.json) to allow BERT model return hidden states
        bert_config = BertConfig.from_pretrained(config.BASE_MODEL_PATH, output_hidden_states=True)

        bert_as_encoder = TFBertModel.from_pretrained(config.BASE_MODEL_PATH, config=bert_config)  # num_labels = len(config.UNIQUE_TAGS)

        # input layers for BERT
        input_ids = Input(shape=(config.MAX_LEN,), name='input_ids', dtype='int32')
        attention_mask = Input(shape=(config.MAX_LEN,), name='attention_mask', dtype='int32')
        token_type_ids = Input(shape=(config.MAX_LEN,), name='token_type_ids', dtype='int32')

        # BERT as embedding 
        embedding = bert_as_encoder(input_ids=input_ids, 
                                    attention_mask=attention_mask, 
                                    token_type_ids=token_type_ids,
                                    return_dict=True,
                                    output_hidden_states=True)

        # downstream model
        if len(config.HIDDEN_STATE_LIST)>=2:
            concat = Concatenate(axis=-1, name='concat')([embedding.hidden_states[x] for x in config.HIDDEN_STATE_LIST]) 
            dropout = Dropout(config.DROPOUT_RATE, name='dropout')(concat)
        else:
            dropout = Dropout(config.DROPOUT_RATE, name='dropout')(embedding.hidden_states[config.HIDDEN_STATE_LIST[0]])

        bilstm = Bidirectional(LSTM(units=config.BILSTM_UNITS, return_sequences=True, recurrent_dropout=config.RECURRENT_DROPOUT_RATE), name='bilstm')(dropout)
        dense = TimeDistributed(Dense(units=len(config.UNIQUE_TAGS), activation="relu"), name='t_dense')(bilstm)  

        # init model
        base = Model(inputs=[input_ids, attention_mask, token_type_ids], outputs = [dense])

        # freeze BERT or not
        base.layers[3].trainable = not config.FREEZE_WHOLE_BERT  # BERT is the 3th layer in 'base'

        # freeze specific BERT encoder layers
        if len(config.FREEZE_BERT_LAYERS_LIST)>=1:
            for i in config.FREEZE_BERT_LAYERS_LIST:
                base.layers[3]._layers[0]._layers[1]._layers[0][i].trainable = False 
                # layers[3]._layers[0]._layers[1]._layers[0] represents 12 layers of BERT

        if summary:
            base.summary()

        # connect CRF
        model = CRFModel(base, len(config.UNIQUE_TAGS))
        model.build([tf.TensorShape([None, config.MAX_LEN]), tf.TensorShape([None, config.MAX_LEN]), tf.TensorShape([None, config.MAX_LEN])])

        if not do_train:
            return model, mirrored_strategy
        
        metric = SparseCategoricalAccuracy('accuracy')
        optimizer, lr_schedule = optimization_tf.create_optimizer(init_lr=config.INIT_LR, 
                                                                   num_train_steps=train_steps,
                                                                   num_warmup_steps=config.NUM_WARMUP_STEPS,
                                                                   min_lr_ratio=config.MIN_LR_RATIO,
                                                                   weight_decay_rate=config.WEIGHT_DECAY_RATE,
                                                                   power=config.POWER
                                                                  )        
        model.compile(optimizer=optimizer, metrics=['accuracy'])
    
        if summary:
            model.summary()

        return model, mirrored_strategy

