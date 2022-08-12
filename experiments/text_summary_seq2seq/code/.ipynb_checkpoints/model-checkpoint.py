import tensorflow as tf
from config import config
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Bidirectional
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

def seq2seq(embedding_matrix):

    encoder_inputs = Input(shape=(config.MAX_LEN_ENCODER,), name='encoder_inputs')
    encoder_embedding = Embedding(
        input_dim=embedding_matrix.vectors.shape[0],    
        output_dim=embedding_matrix.vectors.shape[1],   
        weights=[embedding_matrix.vectors],    
        input_length=config.MAX_LEN_ENCODER,
        trainable=False,
        name='encoder_embedding'
    )(encoder_inputs)  
    encoder_lstm = LSTM(config.HIDDEN_STATES_ENCODER, return_state=True, name='encoder_lstm')
    _, state_h, state_c = encoder_lstm(encoder_embedding)     
    encoder_states = [state_h, state_c]  

    decoder_inputs = Input(shape=(config.MAX_LEN_DECODER,), name='decoder_inputs')
    decoder_embedding = Embedding(
        input_dim=embedding_matrix.vectors.shape[0],    
        output_dim=embedding_matrix.vectors.shape[1],   
        weights=[embedding_matrix.vectors],    
        input_length=config.MAX_LEN_DECODER,
        trainable=False,
        name='decoder_embedding'
    )(decoder_inputs)  
    decoder_lstm = LSTM(config.HIDDEN_STATES_DECODER, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(embedding_matrix.vectors.shape[0], activation="softmax", name='decoder_dense'))
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    metric = SparseCategoricalAccuracy()
    loss = SparseCategoricalCrossentropy()
    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    
    return model


def rebuild_encoder_decoder(model):
    
    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = model.get_layer('encoder_lstm').output          # encoder_lstm
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)
        
    decoder_input_layer = model.input[1] 
    decoder_state_input_h = Input(shape=(config.HIDDEN_STATES_ENCODER,), name='decoder_state_input_h')
    decoder_state_input_c = Input(shape=(config.HIDDEN_STATES_ENCODER,), name='decoder_state_input_c')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_embedding_layer = model.get_layer('decoder_embedding')
    decoder_input_embedded = decoder_embedding_layer(decoder_input_layer)
    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_input_embedded, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.get_layer('time_distributed')
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_input_layer] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model