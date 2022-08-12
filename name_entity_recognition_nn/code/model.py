import tensorflow as tf
from tensorflow.keras.layers import Dense, TimeDistributed, Bidirectional, LSTM, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from keras_crf import CRFModel
import config

def create_model(vocab_size):
    
    input_layer = tf.keras.Input(shape=(config.MAX_LEN,))

    embedding = tf.keras.layers.Embedding(
        vocab_size, 
        config.EMBEDDING_SIZE, 
        mask_zero=True
    )(input_layer)              # batch_size, max_len, emd_size

    mask = Masking()(embedding)

    bilstm1 = Bidirectional(
        LSTM(units=config.BILSTM_UNITS, return_sequences=True, recurrent_dropout=config.RECURRENT_DROPOUT_RATE)
    )(mask)

    bilstm2 = Bidirectional(
        LSTM(units=config.BILSTM_UNITS, return_sequences=True, recurrent_dropout=config.RECURRENT_DROPOUT_RATE)
    )(bilstm1)

    bilstm3 = Bidirectional(
        LSTM(units=config.BILSTM_UNITS, return_sequences=True, recurrent_dropout=config.RECURRENT_DROPOUT_RATE)
    )(bilstm2)

    dense = TimeDistributed(Dense(units=config.DENSE_UNITS, activation="relu"))(bilstm3)  

    base = Model(inputs=input_layer, outputs=dense)

    model = CRFModel(base, len(config.UNIQUE_TAGS))

    optimizer = Adam(learning_rate=config.LEARNING_RATE)

    model.compile(optimizer=optimizer, metrics=['acc'])
    
    return model



