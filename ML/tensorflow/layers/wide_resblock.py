import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import os
from layers.mish import mish
from layers.swish import hard_swish as swish  
def swish_or_mish(se, use_swish=True):
    if use_swish:
        se = Lambda(lambda x: swish(x))(se)
    else:
        se = Lambda(lambda x: mish(x))(se)
    return se

def res_block(init, nb_filter, k=1, use_swish=True):
    x = swish_or_mish(init, use_swish=use_swish)
    x = Conv2D(nb_filter * k, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = swish_or_mish(x, use_swish=use_swish)
    x = Dropout(0.4)(x)
    x = Conv2D(nb_filter * k, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Squeeze_excitation_layer(x, use_swish=use_swish)
    x = add([init, x])
    return x
    
def Squeeze_excitation_layer(input_x, use_swish=True):
    ratio = 4
    out_dim =  input_x.shape[-1]
    squeeze = GlobalAveragePooling2D()(input_x)
    excitation = Dense(units=int(out_dim / ratio))(squeeze)
    excitation = swish_or_mish(excitation, use_swish=use_swish)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape([-1,1,out_dim])(excitation)
    scale = multiply([input_x, excitation])

    return scale
