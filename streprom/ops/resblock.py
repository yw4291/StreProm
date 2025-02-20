import tensorflow as tf
from .conv1d import Conv1D
def ResBlock(inputs, DIM, kernel_size, name):
    output = inputs
    output = tf.nn.leaky_relu(output)
    output = Conv1D(name+'.1', DIM, DIM, kernel_size, output)
    #output = BatchNormalization()(output)
    output = tf.nn.leaky_relu(output)
    output = Conv1D(name+'.2', DIM, DIM, kernel_size, output)
    #output = BatchNormalization()(output)
    return inputs + output

# from keras.layers import *
# def ResBlock(inputs, DIM, kernel_size, name):
#     output = inputs
#     output = tf.nn.leaky_relu(output)
#     #BN1:
#     output = BatchNormalization()(output)
#     output = Conv1D(DIM, kernel_size,padding='same')(output)
#     output = tf.nn.leaky_relu(output)
#     #BN2:
#     output = BatchNormalization()(output)
#     output = Conv1D(DIM, kernel_size,padding='same')(output)
#     return inputs + output