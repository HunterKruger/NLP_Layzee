from config import config
import tensorflow as tf
import transformers
from transformers import TFBertModel, BertConfig    
    
def create_model(do_train=True, train_steps=None, phase=1):    
      
    # must load config file (.json) to allow BERT model return hidden states
    bert_config = BertConfig.from_pretrained(config.BASE_MODEL_PATH, output_hidden_states=True)

    bert_as_encoder = TFBertModel.from_pretrained(config.BASE_MODEL_PATH, config=bert_config, name='bert')  

    # input layers for BERT
    input_ids = tf.keras.layers.Input(shape=(config.MAX_LEN,), name='input_ids', dtype='int32')
    attention_mask = tf.keras.layers.Input(shape=(config.MAX_LEN,), name='attention_mask', dtype='int32')
    token_type_ids = tf.keras.layers.Input(shape=(config.MAX_LEN,), name='token_type_ids', dtype='int32')

    # bert_model
    embedding = bert_as_encoder(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                return_dict=True,
                                output_hidden_states=True)

    ### downstream model
    if len(config.HIDDEN_STATE_LIST) > 1:
        maxpool_list = []
        for i in range(len(config.HIDDEN_STATE_LIST)):
            maxpool = tf.keras.layers.GlobalMaxPooling1D(name='maxpool_'+str(i))(embedding.hidden_states[i])
            maxpool_list.append(maxpool)
        concat = tf.keras.layers.Concatenate(name='concat')(maxpool_list)  
        # get the 0st token's hidden state then concatenate
        dropout = tf.keras.layers.Dropout(config.DROPOUT_RATE, name='dropout')(concat)
    else:
        maxpool = tf.keras.layers.GlobalMaxPooling1D(name='maxpool')(embedding.hidden_states[config.HIDDEN_STATE_LIST[0]])
        dropout = tf.keras.layers.Dropout(config.DROPOUT_RATE, name='dropout')(maxpool)

    output = tf.keras.layers.Dense(config.CLASSES, activation='softmax', name='output')(dropout)  # output layer for multi-clf
    model = tf.keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=output)

    # freeze BERT or not
    model.get_layer('bert').trainable = not config.FREEZE_WHOLE_BERT  # BERT is the 3th layer in the whole model

    # freeze specific BERT encoder layers
    to_freeze_list = config.FREEZE_BERT_LAYERS_LIST if phase==1 else config.FREEZE_BERT_LAYERS_LIST2
    if len(to_freeze_list) >= 1:
        for i in to_freeze_list:
            model.get_layer('bert')._layers[0]._layers[1]._layers[0][i].trainable = False     

    if not do_train:
        return model

    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    if phase == 1:
        optimizer, lr_schedule = transformers.optimization_tf.create_optimizer(
            init_lr=config.INIT_LR, 
            num_train_steps=train_steps,
            num_warmup_steps=config.NUM_WARMUP_STEPS,
            min_lr_ratio=config.MIN_LR_RATIO,
            weight_decay_rate=config.WEIGHT_DECAY_RATE,
            power=config.POWER
            )
    if phase == 2:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            config.SGD_LR,
            decay_steps=config.SGD_DECAY_STEPS,
            decay_rate=config.SGD_DECAY_RATE
        )
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr_schedule, 
            momentum=config.SGD_MOMENTUM, 
            nesterov=config.SGD_NESTEROV
        )

    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    return model
