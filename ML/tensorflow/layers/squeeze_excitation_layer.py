from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

def squeeze_excite_block(input, ratio=16):
    init = input
    filters = input.shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = multiply([init, se])
    return x


def spatial_squeeze_excite_block(input):
    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal')(input)
    x = multiply([input, se])
    return x


def channel_spatial_squeeze_excite(input, ratio=16):
    cse = squeeze_excite_block(input, ratio)
    sse = spatial_squeeze_excite_block(input)
    x = add([cse, sse])
    return x

