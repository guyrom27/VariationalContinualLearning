import numpy as np
import tensorflow as tf
import keras
from generative.load_classifier import load_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)
mnist_cla = load_model("mnist", '../generative/classifier/save/')

def eval_for_torch_data(x_gen, task, sample_per_iter=100):
    x_gen = tf.convert_to_tensor(x_gen.numpy())
    y_gen = tf.clip_by_value(mnist_cla(x_gen), 1e-9, 1.0)
    y_true = np.zeros([sample_per_iter, 10])
    y_true[:, task] = 1
    y_true = tf.constant(np.asarray(y_true, dtype='f'))
    kl = -tf.reduce_sum(y_true * tf.log(y_gen), 1)
    kl_mean = tf.reduce_mean(kl)
    kl_var = tf.reduce_mean((kl - kl_mean) ** 2)

    ops = [kl_mean, kl_var]
    kl_mean, kl_var = sess.run(ops)

    return kl_mean, kl_var
