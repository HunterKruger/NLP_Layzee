import config
import tensorflow as tf
import transformers
from transformers import TFBertModel, BertConfig
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy

    
def create_model(do_train=True, train_steps=None):    
        
    # must load config file (.json) to allow BERT model return hidden states
    bert_config = BertConfig.from_pretrained(config.BASE_MODEL_PATH, output_hidden_states=True)
    bert_layer = TFBertModel.from_pretrained(config.BASE_MODEL_PATH, config=bert_config, name='bert')  

    # input layers for BERT
    input_ids = Input(shape=(config.MAX_LEN,), name='input_ids', dtype='int32')
    attention_mask = Input(shape=(config.MAX_LEN,), name='attention_mask', dtype='int32')
    token_type_ids = Input(shape=(config.MAX_LEN,), name='token_type_ids', dtype='int32')

    # bert_model
    embedding = bert_layer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        return_dict=True,
        output_hidden_states=True
    )

    ### downstream model
    start_logits = Dense(1, name="start_logit", use_bias=False)(embedding[0])
    start_logits = Flatten()(start_logits)
    end_logits = Dense(1, name="end_logit", use_bias=False)(embedding[0])
    end_logits = Flatten()(end_logits)
    start_probs = Activation('softmax')(start_logits)
    end_probs = Activation('softmax')(end_logits)
    model = Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=[start_probs, end_probs])

    # freeze BERT or not
    model.get_layer('bert').trainable = not config.FREEZE_WHOLE_BERT  # BERT is the 3th layer in the whole model

    # freeze specific BERT encoder layers
    to_freeze_list = config.FREEZE_BERT_LAYERS_LIST
    if len(to_freeze_list) >= 1:
        for i in to_freeze_list:
            model.get_layer('bert')._layers[0]._layers[1]._layers[0][i].trainable = False    

    if not do_train:
        return model

    metric = SparseCategoricalAccuracy()
    loss = SparseCategoricalCrossentropy(from_logits=False)
    optimizer, _ = transformers.optimization_tf.create_optimizer(
        init_lr=config.INIT_LR, 
        num_train_steps=train_steps,
        num_warmup_steps=config.NUM_WARMUP_STEPS,
        min_lr_ratio=config.MIN_LR_RATIO,
        weight_decay_rate=config.WEIGHT_DECAY_RATE,
        power=config.POWER
    )
    model.compile(loss=[loss, loss], optimizer=optimizer, metrics=metric)

    return model

