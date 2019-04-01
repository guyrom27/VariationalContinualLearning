
import tensorflow as tf
import numpy as np

def init_weights(input_size, output_size, constant=1.0, seed=123): 
    """ Glorot and Bengio, 2010's initialization of network weights"""
    scale = constant*np.sqrt(6.0/(input_size + output_size))
    if output_size > 0:
        return tf.random_uniform((input_size, output_size), 
                             minval=-scale, maxval=scale, 
                             dtype=tf.float32, seed=seed)
    else:
        return tf.random_uniform([input_size], 
                             minval=-scale, maxval=scale, 
                             dtype=tf.float32, seed=seed)

class mlp_layer:

    def __init__(self, d_in, d_out, activation, name):
        self.W = tf.Variable(init_weights(d_in, d_out), name = name+'_W')
        self.b = tf.Variable(tf.zeros([d_out]), name = name+'_b')
        self.activation = activation
    
    def __call__(self, x):
        a = tf.matmul(x, self.W) + self.b
        if self.activation == 'relu':
            return tf.nn.relu(a)
        if self.activation == 'sigmoid':
            return tf.nn.sigmoid(a)
        if self.activation == 'linear':
            return a  


