import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

def simple_swish(features):
  return features * K.sigmoid(features)

def hard_swish(features):
  """Computes a hard version of the swish function.
  This operation can be used to reduce computational cost and improve
  quantization for edge devices.
  Args:
    features: A `Tensor` representing preactivation values.
  Returns:
    The activation value.
  """
  return features * tf.nn.relu6(features + tf.constant(3.)) * (1. / 6.)


