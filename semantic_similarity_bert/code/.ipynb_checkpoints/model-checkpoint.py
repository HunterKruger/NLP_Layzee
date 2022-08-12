from config import config
import tensorflow as tf
import transformers
from transformers import TFBertModel, BertConfig
from tensorflow.keras import backend as K

def focal_loss(y_true, y_pred, gamma = 2.0, alpha = 0.8):
    #https://zhuanlan.zhihu.com/p/103623160
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return - K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def create_model(do_train=True, train_steps=None, summary=True):
    
    # must load config file (.json) to allow BERT model return hidden states
    bert_config = BertConfig.from_pretrained(config.BASE_MODEL_PATH, output_hidden_states=True)

    bert_as_encoder = TFBertModel.from_pretrained(config.BASE_MODEL_PATH, config = bert_config) 

    # input layers for BERT
    input_ids = tf.keras.layers.Input(shape=(config.MAX_LEN,), name='input_ids', dtype='int32')
    attention_mask = tf.keras.layers.Input(shape=(config.MAX_LEN,), name='attention_mask', dtype='int32')
    token_type_ids = tf.keras.layers.Input(shape=(config.MAX_LEN,), name='token_type_ids', dtype='int32')

    # bert_model: returns (sequence_output, pooled_output)
    embedding = bert_as_encoder(input_ids = input_ids, 
                                attention_mask = attention_mask, 
                                token_type_ids = token_type_ids,
                                return_dict = True,
                                output_hidden_states=True)

    ### downstream model

    # input: (batch_size, max_len, hidden_state_size)
    max_pool = tf.keras.layers.GlobalMaxPooling1D(name='maxpool')(embedding.hidden_states[-1])  
    avg_pool = tf.keras.layers.GlobalAveragePooling1D(name='avgpool')(embedding.hidden_states[-1]) 
    concat = tf.keras.layers.concatenate([avg_pool, max_pool], name='concat')
    dense = tf.keras.layers.Dense(config.DENSE_UNITS, activation='relu', name='dense')(concat)
    dropout = tf.keras.layers.Dropout(config.DROPOUT_RATE, name='dropout')(dense)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(dropout)   # output layer for bin-clf
    model = tf.keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs = [output])


    # freeze whole BERT or not
    model.layers[3].trainable = not config.FREEZE_WHOLE_BERT  # BERT is the 3th layer in the whole model

    # freeze specific BERT layers
    if len(config.FREEZE_BERT_LAYERS_LIST)>=1:
        for i in config.FREEZE_BERT_LAYERS_LIST:
            model.layers[3]._layers[0]._layers[1]._layers[0][i].trainable = False 
            # layers[3]._layers[0]._layers[1]._layers[0] represents 12 layers of BERT

    if not do_train:
        return model

    metric = tf.keras.metrics.BinaryAccuracy('accuracy')
    optimizer, lr_schedule = transformers.optimization_tf.create_optimizer(init_lr=config.INIT_LR, 
                                                                           num_train_steps=train_steps,
                                                                           num_warmup_steps=config.NUM_WARMUP_STEPS,
                                                                           min_lr_ratio=config.MIN_LR_RATIO,
                                                                           weight_decay_rate=config.WEIGHT_DECAY_RATE,
                                                                           power=config.POWER
                                                                          )

    model.compile(loss=[focal_loss], optimizer=optimizer, metrics=['accuracy'])

    if summary:
        model.summary()

    return model



