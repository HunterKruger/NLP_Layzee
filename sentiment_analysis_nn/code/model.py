from tensorflow.python.keras.layers.core import Masking
import config
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Flatten, Concatenate, Dense, Bidirectional,LSTM, Masking
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

def create_model_cnn(embedding_matrix, do_train=True):    

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
    dense = Dense(config.CLASSES, activation='softmax')(concat)
    model = Model(inputs=input_layer, outputs=dense)
    
    if not do_train:
        return model

    metric = SparseCategoricalAccuracy()
    loss = SparseCategoricalCrossentropy()
    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    return model


def create_model_rnn(embedding_matrix, do_train=True):  

    input_layer = Input(shape=(config.MAX_LEN,))

    embedding = Embedding(
        input_dim=embedding_matrix.vectors.shape[0],    
        output_dim=embedding_matrix.vectors.shape[1],   
        weights=[embedding_matrix.vectors], 
        input_length=config.MAX_LEN,
        trainable=False
    )(input_layer)               # no need to set mask_zero to True cuz PAD's embedding has been set to a zero vec
    
    mask = Masking()(embedding)  # mask timestep which is all zero
    bilstm = Bidirectional(LSTM(config.BILSTM_UNITS, return_sequences=True))(mask)
    maxpool = GlobalMaxPooling1D()(bilstm)
    dense = Dense(config.CLASSES, activation='softmax')(maxpool)

    model = Model(inputs=input_layer, outputs=dense)

    if not do_train:
        return model

    metric = SparseCategoricalAccuracy()
    loss = SparseCategoricalCrossentropy()
    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    return model

