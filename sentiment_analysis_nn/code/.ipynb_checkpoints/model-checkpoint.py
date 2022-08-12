from config import config
import tensorflow as tf

    
def create_model_cnn(embedding_matrix, do_train=True):    

    input_layer = tf.keras.Input(shape=(config.MAX_LEN,))
    embedding = tf.keras.layers.Embedding(
        input_dim=embedding_matrix.vectors.shape[0],    
        output_dim=embedding_matrix.vectors.shape[1],   
        weights=[embedding_matrix.vectors], 
        input_length=config.MAX_LEN, 
        trainable=False
    )(input_layer)

    flattens = []
    for i in range(len(config.FILTERS)):
        cnn = tf.keras.layers.Conv1D(
            filters=config.FILTERS[i], 
            kernel_size=(config.KERNEL_SIZE[i],),
            activation='relu',
            padding="valid",
            strides=config.STRIDES[i]
           )(embedding)
        maxpool = tf.keras.layers.GlobalMaxPooling1D()(cnn)
        flatten = tf.keras.layers.Flatten()(maxpool)
        flattens.append(flatten)

    concat = tf.keras.layers.Concatenate(axis=-1)(flattens)
    dense = tf.keras.layers.Dense(config.CLASSES, activation='softmax')(concat)
    model = tf.keras.models.Model(inputs=input_layer, outputs=dense)
    
    if not do_train:
        return model

    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    return model


def create_model_rnn(embedding_matrix, do_train=True):  

    input_layer = tf.keras.Input(shape=(config.MAX_LEN,))

    embedding = tf.keras.layers.Embedding(
        input_dim=embedding_matrix.vectors.shape[0],    
        output_dim=embedding_matrix.vectors.shape[1],   
        weights=[embedding_matrix.vectors], 
        input_length=config.MAX_LEN,
        trainable=False
    )(input_layer)
    
    bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.BILSTM_UNITS, return_sequences=True))(embedding)
    maxpool = tf.keras.layers.GlobalMaxPooling1D()(bilstm)
    dense = tf.keras.layers.Dense(config.CLASSES, activation='softmax')(maxpool)
    model = tf.keras.models.Model(inputs=input_layer, outputs=dense)

    if not do_train:
        return model

    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    return model

