import config
import tensorflow as tf
import transformers
from transformers import TFBertModel, BertConfig
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GlobalMaxPooling1D, Concatenate, Dense, GlobalAveragePooling1D, BatchNormalization, Lambda, Dropout
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import BinaryAccuracy, AUC


def focal_loss(gamma=2.0, alpha=0.8):   
    '''
    Focal loss for binary classification.
    gamma: difficulty weight, 0 to disable, try in range(0.5, 10.0)
    alpha: class_weight, 0.5 to disable, if positive_class:negative_class=20:80, use alpha=0.80
    https://zhuanlan.zhihu.com/p/103623160
    '''
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return - K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.
    Arguments:
        vects: List containing two tensors of same length.
    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, K.epsilon()))  
    # epsilon():  Epsilon is small value (1e-07 in TensorFlow Core v2.2.0) that makes very little difference to the value of the denominator, but ensures that it isn't equal to exactly zero
    # reduce_sum(): Computes the sum of elements across dimensions of a tensor.
    # square(): element-wise square 
    # sqrt(): element-with root square


def constrastive_loss(margin=1): 
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.
    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).
    Returns:
        'constrastive_loss' function with data ('margin') attached.
    """
    # Contrastive loss = mean((1-true_value) * square(prediction) + true_value * square(max(margin-prediction, 0)))
    def contrastive_loss_fixed(y_true, y_pred):
        """Calculates the constrastive loss.
        Arguments:
        y_true: List of labels, each label is of type float32.
        y_pred: List of predictions of same length as of y_true,
                each label is of type float32.
        Returns:
                A tensor containing constrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)
    return contrastive_loss_fixed


def create_model(do_train=True, train_steps=None):   
    '''
    Single BERT
    '''
    
    # must load config file (.json) to allow BERT model return hidden states
    bert_config = BertConfig.from_pretrained(config.BASE_MODEL_PATH, output_hidden_states=True)
    bert_as_encoder = TFBertModel.from_pretrained(config.BASE_MODEL_PATH, config=bert_config, name='bert') 

    # input layers for BERT
    input_ids = Input(shape=(config.MAX_LEN,), name='input_ids', dtype='int32')
    attention_mask = Input(shape=(config.MAX_LEN,), name='attention_mask', dtype='int32')
    token_type_ids = Input(shape=(config.MAX_LEN,), name='token_type_ids', dtype='int32')

    # bert_model: returns (sequence_output, pooled_output)
    embedding = bert_as_encoder(
        input_ids = input_ids, 
        attention_mask = attention_mask, 
        token_type_ids = token_type_ids,
        return_dict = True,
        output_hidden_states=True
    )

    ### downstream model
    # input: (batch_size, max_len, hidden_state_size)
    max_pool = GlobalMaxPooling1D(name='maxpool')(embedding.hidden_states[-1])  
    avg_pool = GlobalAveragePooling1D(name='avgpool')(embedding.hidden_states[-1]) 
    concat = Concatenate(name='concat')([avg_pool, max_pool])
    dense = Dense(config.DENSE_UNITS, activation='relu', name='dense')(concat)
    dropout = Dropout(config.DROPOUT_RATE, name='dropout')(dense)
    output = Dense(1, activation='sigmoid', name='output')(dropout)   # output layer for bin-clf
    model = Model(inputs=[input_ids, attention_mask, token_type_ids], outputs = output)

    # freeze BERT or not
    model.get_layer('bert').trainable = not config.FREEZE_WHOLE_BERT  # BERT is the 3th layer in the whole model

    # freeze specific BERT encoder layers
    to_freeze_list = config.FREEZE_BERT_LAYERS_LIST
    if len(to_freeze_list) >= 1:
        for i in to_freeze_list:
            model.get_layer('bert')._layers[0]._layers[1]._layers[0][i].trainable = False    

    if not do_train:
        return model

    optimizer, _ = transformers.optimization_tf.create_optimizer(
        init_lr=config.INIT_LR, 
        num_train_steps=train_steps,
        num_warmup_steps=config.NUM_WARMUP_STEPS,
        min_lr_ratio=config.MIN_LR_RATIO,
        weight_decay_rate=config.WEIGHT_DECAY_RATE,
        power=config.POWER
    )

    metric = BinaryAccuracy()
    metric2 = AUC()
    model.compile(loss=focal_loss(), optimizer=optimizer, metrics=[metric, metric2])
    model.summary()

    return model


def create_base_model():   
    '''
    BERT in Siamese
    '''
    
    # must load config file (.json) to allow BERT model return hidden states
    bert_config = BertConfig.from_pretrained(config.BASE_MODEL_PATH, output_hidden_states=True)
    bert_as_encoder = TFBertModel.from_pretrained(config.BASE_MODEL_PATH, config=bert_config, name='bert')  

    # input layers for BERT
    input_ids = Input(shape=(config.MAX_LEN_SIAMESE,), name='input_ids', dtype='int32')
    attention_mask = Input(shape=(config.MAX_LEN_SIAMESE,), name='attention_mask', dtype='int32')
    token_type_ids = Input(shape=(config.MAX_LEN_SIAMESE,), name='token_type_ids', dtype='int32')

    # bert_model: returns (sequence_output, pooled_output)
    embedding = bert_as_encoder(
        input_ids = input_ids, 
        attention_mask = attention_mask, 
        token_type_ids = token_type_ids,
        return_dict = True,
        output_hidden_states=True
    )

    ### downstream model

    # input: (batch_size, max_len, hidden_state_size)
    avg_pool = GlobalAveragePooling1D(name='avgpool')(embedding.hidden_states[-1]) 
    model = Model(inputs=[input_ids, attention_mask, token_type_ids], outputs = avg_pool)

    # freeze BERT or not
    model.get_layer('bert').trainable = not config.FREEZE_WHOLE_BERT  # BERT is the 3th layer in the whole model

    # freeze specific BERT encoder layers
    to_freeze_list = config.FREEZE_BERT_LAYERS_LIST
    if len(to_freeze_list) >= 1:
        for i in to_freeze_list:
            model.get_layer('bert')._layers[0]._layers[1]._layers[0][i].trainable = False    
    
    model.summary()

    return model


def create_siamese_model(do_train=True, train_steps=None):    
    '''
    Siamese
    '''

    base_network = create_base_model()

    input_a = [Input(shape=(config.MAX_LEN_SIAMESE,)),Input(shape=(config.MAX_LEN_SIAMESE,)),Input(shape=(config.MAX_LEN_SIAMESE,))]
    input_b = [Input(shape=(config.MAX_LEN_SIAMESE,)),Input(shape=(config.MAX_LEN_SIAMESE,)),Input(shape=(config.MAX_LEN_SIAMESE,))]
    
    output_a = base_network(input_a)
    output_b = base_network(input_b)

    distance = Lambda(euclidean_distance)([output_a, output_b])
    normalization = BatchNormalization()(distance)
    output = Dense(units=1, activation='sigmoid')(normalization)
    model = Model(inputs=[input_a, input_b], outputs=output)

    if not do_train:
        return model

    optimizer, _ = transformers.optimization_tf.create_optimizer(
        init_lr=config.INIT_LR, 
        num_train_steps=train_steps,
        num_warmup_steps=config.NUM_WARMUP_STEPS,
        min_lr_ratio=config.MIN_LR_RATIO,
        weight_decay_rate=config.WEIGHT_DECAY_RATE,
        power=config.POWER
    )

    metric = BinaryAccuracy()
    metric2 = AUC()

    model.compile(loss=constrastive_loss(margin=1), optimizer=optimizer, metrics=[metric, metric2])
    
    model.summary()

    return model

