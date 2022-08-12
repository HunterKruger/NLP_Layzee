from config import config
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Flatten, Concatenate, Dense, Bidirectional,LSTM, GlobalAveragePooling1D, BatchNormalization, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from tensorflow.keras.optimizers import Adam


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
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


def loss(margin=config.MARGIN):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'constrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
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

    return contrastive_loss


def create_model_cnn(embedding_matrix):    

    input_layer = Input(shape=(config.MAX_LEN,))
    embedding = Embedding(
        input_dim=embedding_matrix.vectors.shape[0],    
        output_dim=embedding_matrix.vectors.shape[1],   
        weights=[embedding_matrix.vectors], 
        input_length=config.MAX_LEN, 
        trainable=False
    )(input_layer)

    flattens = []
    for i in range(len(config.FILTERS)):
        cnn = Conv1D(
            filters=config.FILTERS[i], 
            kernel_size=(config.KERNEL_SIZE[i],),
            activation='relu',
            padding="valid",
            strides=config.STRIDES[i]
           )(embedding)
        maxpool = GlobalMaxPooling1D()(cnn)
        flatten = Flatten()(maxpool)
        flattens.append(flatten)

    concat = Concatenate(axis=-1)(flattens)
    model = Model(inputs=input_layer, outputs=concat)
    
    model.summary()

    return model


def create_model_rnn(embedding_matrix):  

    input_layer = Input(shape=(config.MAX_LEN,))

    embedding = Embedding(
        input_dim=embedding_matrix.vectors.shape[0],    
        output_dim=embedding_matrix.vectors.shape[1],   
        weights=[embedding_matrix.vectors], 
        input_length=config.MAX_LEN,
        trainable=False
    )(input_layer)
    
    bilstm = Bidirectional(LSTM(config.BILSTM_UNITS, return_sequences=True))(embedding)
    avgpool = GlobalAveragePooling1D()(bilstm)
    model = Model(inputs=input_layer, outputs=avgpool)
    
    model.summary()

    return model


def create_siamese_model(embedding_matrix, use_rnn=config.USE_RNN, fake_siamese=config.FAKE_SIAMESE):

    if use_rnn:
        base_network = create_model_rnn(embedding_matrix)
        if fake_siamese:
            base_network2 = create_model_rnn(embedding_matrix)
    else:
        base_network = create_model_cnn(embedding_matrix)
        if fake_siamese:
            base_network2 = create_model_cnn(embedding_matrix)

    input_a = tf.keras.Input(shape=(config.MAX_LEN,))
    input_b = tf.keras.Input(shape=(config.MAX_LEN,))
    
    output_a = base_network(input_a)
    if fake_siamese:
        output_b = base_network2(input_b)
    else:
        output_b = base_network(input_b)

    distance = Lambda(euclidean_distance)([output_a, output_b])
    normalization = BatchNormalization()(distance)
    output = Dense(units=1, activation='sigmoid')(normalization)
    model = Model(inputs=[input_a, input_b], outputs=output)

    metric = BinaryAccuracy()
    metric2 = AUC()

    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    model.compile(loss=loss(margin=1), optimizer=optimizer, metrics=[metric, metric2])
    
    model.summary()

    return model

