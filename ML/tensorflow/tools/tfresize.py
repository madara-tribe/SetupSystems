import tensorflow.keras

def tfresize(x, size=7):
    return Lambda(lambda x: tensorflow.keras.image.resize(x, (size, size)))(x)
