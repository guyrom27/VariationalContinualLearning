import numpy as np
import tensorflow as tf
from mlp import init_weights

"""
A Bayesian MLP generator
"""

def sample_gaussian(mu, log_sig):
    return mu + tf.exp(log_sig) * tf.random_normal(mu.get_shape())

class bayesian_mlp_layer:

    def __init__(self,d_in, d_out, activation, name):
        self.mu_W = tf.Variable(init_weights(d_in, d_out), name = name+'_mu_W')
        self.mu_b = tf.Variable(tf.zeros([d_out]), name = name+'_mu_b')
        self.log_sig_W = tf.Variable(tf.ones([d_in, d_out])*-6, name = name+'_log_sig_W')
        self.log_sig_b = tf.Variable(tf.ones([d_out])*-6, name = name+'_log_sig_b')
        self.activation = activation

    def __call__(self, x, sampling=True):
        if sampling:
            W = sample_gaussian(self.mu_W, self.log_sig_W)
            b = sample_gaussian(self.mu_b, self.log_sig_b)
        else:
            print('use mean of q(theta)...')
            W = self.mu_W; b = self.mu_b
        a = tf.matmul(x, W) + b
        if self.activation == 'relu':
            return tf.nn.relu(a)
        if self.activation == 'sigmoid':
            return tf.nn.sigmoid(a)
        if self.activation == 'linear':
            return a  
            

def generator_head(dimZ, dimH, n_layers, name):
    fc_layer_sizes = [dimZ] + [dimH for i in range(n_layers)]
    layers = []
    N_layers = len(fc_layer_sizes) - 1
    for i in range(N_layers):
        d_in = fc_layer_sizes[i]; d_out = fc_layer_sizes[i+1]
        name_layer = name + '_head_l%d' % i
        layers.append(bayesian_mlp_layer(d_in, d_out, 'relu', name_layer))
    
    print('decoder head MLP of size', fc_layer_sizes)
    
    def apply(x, sampling=True):
        for layer in layers:
            x = layer(x, sampling)
        return x
        
    return apply

class generator_shared:

    def __init__(self,dimX, dimH, n_layers, last_activation, name):
        # now construct a decoder
        fc_layer_sizes = [dimH for i in range(n_layers)] + [dimX]
        self.layers = []
        self.N_layers = len(fc_layer_sizes) - 1
        for i in range(self.N_layers):
            d_in = fc_layer_sizes[i]; d_out = fc_layer_sizes[i+1]
            if i < self.N_layers - 1:
                activation = 'relu'
            else:
                activation = last_activation
            name_layer = name + '_shared_l%d' % i
            self.layers.append(bayesian_mlp_layer(d_in, d_out, activation, name_layer))

        print('decoder shared MLP of size', fc_layer_sizes)
    
    def __call__(self, x, sampling=True):
        for layer in self.layers:
            x = layer(x, sampling)
        return x

    
class generator:

    def __init__(self, head_net, shared_net):
        self.head_net = head_net
        self.shared_net = shared_net

    def __call__(self, x, sampling=True):
        x = self.head_net(x)
        x = self.shared_net(x)
        return x

def construct_gen(gen, dimZ, sampling=True):
    def gen_data(N):
        # start from sample z_0, generate data
        z = tf.random_normal(shape=(N, dimZ))
        return gen(z, sampling)

    return gen_data
    
