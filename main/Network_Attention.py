from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, Reshape, Multiply, Dense, Softmax
from keras.models import Model

def attention_module(query, keys, channels):
    # Transform query and keys to same feature space
    query = Dense(channels)(query)
    keys = Dense(channels)(keys)

    # Compute attention weights
    attention_scores = Multiply()([query, keys])
    attention_scores = Softmax(axis=-1)(attention_scores)

    # Weighted sum of keys
    weighted_keys = Multiply()([attention_scores, keys])
    weighted_sum = Reshape((1, 1, channels))(weighted_keys)
    return weighted_sum

def build_mbllen(input_shape):

    def EM(input, kernal_size, channel):
        conv_1 = Conv2D(channel, (3, 3), activation='relu', padding='same', data_format='channels_last')(input)
        conv_2 = Conv2D(channel, (kernal_size, kernal_size), activation='relu', padding='valid', data_format='channels_last')(conv_1)
        conv_3 = Conv2D(channel*2, (kernal_size, kernal_size), activation='relu', padding='valid', data_format='channels_last')(conv_2)
        conv_4 = Conv2D(channel*4, (kernal_size, kernal_size), activation='relu', padding='valid', data_format='channels_last')(conv_3)
        conv_5 = Conv2DTranspose(channel*2, (kernal_size, kernal_size), activation='relu', padding='valid', data_format='channels_last')(conv_4)
        conv_6 = Conv2DTranspose(channel, (kernal_size, kernal_size), activation='relu', padding='valid', data_format='channels_last')(conv_5)
        res = Conv2DTranspose(3, (kernal_size, kernal_size), activation='relu', padding='valid', data_format='channels_last')(conv_6)
        return res

    inputs = Input(shape=input_shape)
    FEM = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(inputs)
    EM_com = EM(FEM, 5, 8)

    # Add attention module
    for j in range(3):
        for i in range(0, 3):
            FEM = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(FEM)
            EM1 = EM(FEM, 5, 8)
            attention = attention_module(EM1, EM_com, 64)
            EM_com = Concatenate(axis=3)([EM_com, attention])

    outputs = Conv2D(3, (1, 1), activation='relu', padding='same', data_format='channels_last')(EM_com)
    return Model(inputs, outputs)
