from config import config
import tensorflow as tf
import transformers
from transformers import AutoTokenizer, AutoModel, TFBertModel, BertConfig
from datasets import load_metric
    
def create_model(do_train=True, train_steps=None, summary=True):    
        
    # must load config file (.json) to allow BERT model return hidden states
    bert_config = BertConfig.from_pretrained(config.BASE_MODEL_PATH, output_hidden_states=True)

    bert_layer = TFBertModel.from_pretrained(config.BASE_MODEL_PATH,
                                                  config=bert_config)  # , num_labels = len(config.UNIQUE_TAGS)

    # input layers for BERT
    input_ids = tf.keras.layers.Input(shape=(config.MAX_LEN,), name='input_ids', dtype='int32')
    attention_mask = tf.keras.layers.Input(shape=(config.MAX_LEN,), name='attention_mask', dtype='int32')
    token_type_ids = tf.keras.layers.Input(shape=(config.MAX_LEN,), name='token_type_ids', dtype='int32')

    # bert_model
    embedding = bert_layer(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           return_dict=True,
                           output_hidden_states=True)

    ### downstream model

    start_logits = tf.keras.layers.Dense(1, name="start_logit", use_bias=False)(embedding[0])
    start_logits = tf.keras.layers.Flatten()(start_logits)

    end_logits = tf.keras.layers.Dense(1, name="end_logit", use_bias=False)(embedding[0])
    end_logits = tf.keras.layers.Flatten()(end_logits)

    start_probs = tf.keras.layers.Activation('softmax')(start_logits)
    end_probs = tf.keras.layers.Activation('softmax')(end_logits)

    model = tf.keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=[start_probs, end_probs])

    # freeze BERT or not
    model.layers[3].trainable = not config.FREEZE_WHOLE_BERT  # BERT is the 3th layer in the whole model

    # freeze specific BERT encoder layers
    if len(config.FREEZE_BERT_LAYERS_LIST) >= 1:
        for i in config.FREEZE_BERT_LAYERS_LIST:
            model.layers[3]._layers[0]._layers[1]._layers[0][i].trainable = False     
            #layers[3]._layers[0]._layers[1]._layers[0] represents 12 layers of BERT

    if not do_train:
        return model

    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer, lr_schedule = transformers.optimization_tf.create_optimizer(init_lr=config.INIT_LR, 
                                                                           num_train_steps=train_steps,
                                                                           num_warmup_steps=config.NUM_WARMUP_STEPS,
                                                                           min_lr_ratio=config.MIN_LR_RATIO,
                                                                           weight_decay_rate=config.WEIGHT_DECAY_RATE,
                                                                           power=config.POWER)
    model.compile(loss=[loss, loss], optimizer=optimizer, metrics=metric)

    if summary:
        model.summary()

    return model

