import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer, Input, Embedding, LSTM, Dense, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from config import config


class Encoder(Model):
    
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(Encoder, self).__init__()
        # Embedding Layer
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        # Encode LSTM Layer
        self.encoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True, name="encode_lstm")
        
    def call(self, inputs):
        encoder_embed = self.embedding(inputs)
        encoder_outputs, state_h, state_c = self.encoder_lstm(encoder_embed)
        return encoder_outputs, state_h, state_c
    
    
class Decoder(Model):
    
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(Decoder, self).__init__()
        # Embedding Layer
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        # Decode LSTM Layer
        self.decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True, name="decode_lstm")
        # Attention Layer: Loung attention
        self.attention = Attention()
    
    def call(self, enc_outputs, dec_inputs, states_inputs):
        decoder_embed = self.embedding(dec_inputs)
        dec_outputs, dec_state_h, dec_state_c = self.decoder_lstm(decoder_embed, initial_state=states_inputs)
        attention_output = self.attention([dec_outputs, enc_outputs])
        return attention_output, dec_state_h, dec_state_c
    
    
def seq2seq(maxlen_encoder, maxlen_decoder, embedding_dim, hidden_units, vocab_size):
    """
    seq2seq model
    """
    # Input Layer
    encoder_inputs = Input(shape=(maxlen_encoder,), name="encode_input")
    decoder_inputs = Input(shape=(maxlen_decoder,), name="decode_input")
    # Encoder Layer
    encoder = Encoder(vocab_size, embedding_dim, hidden_units)
    enc_outputs, enc_state_h, enc_state_c = encoder(encoder_inputs)
    dec_states_inputs = [enc_state_h, enc_state_c]
    # Decoder Layer
    decoder = Decoder(vocab_size, embedding_dim, hidden_units)
    attention_output, dec_state_h, dec_state_c = decoder(enc_outputs, decoder_inputs, dec_states_inputs)
    # Dense Layer
    dense_outputs = Dense(vocab_size, activation='softmax', name="dense")(attention_output)
    # seq2seq model
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=dense_outputs)
    
    metric = SparseCategoricalAccuracy()
    loss = SparseCategoricalCrossentropy()
    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    
    return model


def encoder_infer(model):
    encoder_model = Model(inputs=model.get_layer('encoder').input, outputs=model.get_layer('encoder').output)
    return encoder_model


def decoder_infer(model, encoder_model):
    encoder_output = encoder_model.get_layer('encoder').output[0]
    maxlen, hidden_units = encoder_output.shape[1:]
    
    dec_input = model.get_layer('decode_input').input
    enc_output = Input(shape=(maxlen, hidden_units), name='enc_output')
    dec_input_state_h = Input(shape=(hidden_units,), name='input_state_h')
    dec_input_state_c = Input(shape=(hidden_units,), name='input_state_c')
    dec_input_states = [dec_input_state_h, dec_input_state_c]

    decoder = model.get_layer('decoder')
    dec_outputs, out_state_h, out_state_c = decoder(enc_output, dec_input, dec_input_states)
    dec_output_states = [out_state_h, out_state_c]

    decoder_dense = model.get_layer('dense')
    dense_output = decoder_dense(dec_outputs)

    decoder_model = Model(inputs=[enc_output, dec_input, dec_input_states], outputs=[dense_output]+dec_output_states)
    
    return decoder_model
