import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, LSTM, Dense, Dropout, Flatten, Activation, RepeatVector, Permute, Multiply, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

def attention_layer(inputs, time_steps):
    pass
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a) 
    
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def build_caelstm_model(input_shape):
    inputs = Input(shape=input_shape)
    
    e = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(inputs)
    e = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(e)
    e = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(e)
    e = MaxPooling1D(pool_size=2, padding='same')(e)
    
    l = LSTM(128, return_sequences=True)(e)
    
    att = tf.keras.layers.TimeDistributed(Dense(1, activation='tanh'))(l)
    att = Flatten()(att)
    att = Activation('softmax')(att)
    
    att = RepeatVector(128)(att)
    att = Permute((2, 1))(att)
    
    att_out = Multiply()([l, att])
    
    att_out = Lambda(lambda x: K.sum(x, axis=1))(att_out)
    
    d = Dense(200, activation='relu')(att_out)
    d = Dropout(0.4)(d)
    
    d = Dense(100, activation='relu')(d)
    d = Dropout(0.4)(d)
    outputs = Dense(1, activation='linear')(d)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    model = build_caelstm_model((30, 14))
    model.summary()
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    print("Model compiled successfully.")
