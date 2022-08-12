import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Dropout, Bidirectional, LSTM, Concatenate
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from keras_crf import CRFModel

from config import config

def create_model(vocab_size):
    
    input_layer = tf.keras.Input(shape=(config.MAX_LEN,))
    embedding = tf.keras.layers.Embedding(vocab_size,config.EMBEDDING_SIZE)(input_layer)   # batch_size, max_len, emd_size
    
    bilstm1 = Bidirectional(LSTM(units=config.BILSTM_UNITS, return_sequences=True, recurrent_dropout=config.RECURRENT_DROPOUT_RATE))(embedding)
    bilstm2 = Bidirectional(LSTM(units=config.BILSTM_UNITS, return_sequences=True, recurrent_dropout=config.RECURRENT_DROPOUT_RATE))(bilstm1)
    
    dense = TimeDistributed(Dense(units=len(config.UNIQUE_TAGS), activation="relu"))(bilstm2)  
    base = Model(inputs=input_layer, outputs=dense)

    model = CRFModel(base, len(config.UNIQUE_TAGS))

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(optimizer=optimizer, metrics=['acc'])
    
    return model



