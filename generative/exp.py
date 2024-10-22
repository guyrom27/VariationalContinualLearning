import numpy as np
import tensorflow as tf
import keras
import sys, os

sys.stdout = open('coreset_session1004.txt.3',mode='w',buffering=1)
np.random.seed(42)

sys.path.extend(['alg/', 'models/'])
from visualisation import plot_images
from encoder_no_shared import encoder, recon
from utils import init_variables, save_params, load_params, load_data
from alg.eval_test_ll import construct_eval_func
from alg.eval_test_class import construct_eval_func2
from load_classifier import load_model

from bayesian_generator import generator_head, generator_shared, \
            generator, construct_gen
from onlinevi import construct_optimizer, init_shared_prior, \
            update_shared_prior, update_q_sigma, update_headnet_prior

dimZ = 50
dimH = 500
n_channel = 128
batch_size = 50
lr = 1e-4
K_mc = 10
train = False   # Trains a model if True, load parameters otherwise
checkpoint = -1

coreset_size = 100
n_coreset_iter = 33

print_weights = False
degenerate_dataset = False
coreset_session = True

data_path = './classifier/'


def main(data_name, method, dimZ, dimH, n_channel, batch_size, K_mc, checkpoint, lbd):
    # set up dataset specific stuff
    from config import config
    labels, n_iter, dimX, shape_high, ll = config(data_name, n_channel)
    if data_name == 'mnist':
        from mnist import load_mnist
    if data_name == 'notmnist':
        from generative.classifier.notmnist import load_notmnist

    # then define model
    n_layers_shared = 2
    batch_size_ph = tf.placeholder(tf.int32, shape=(), name='batch_size')
    dec_shared = generator_shared(dimX, dimH, n_layers_shared, 'sigmoid', 'gen')

    # initialise sessions
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    string = method
    if method == 'onlinevi' and K_mc > 1:
        string = string + '_K%d' % K_mc
    path_name = data_name + '_%s/' % string
    if not os.path.isdir('save/'):
        os.mkdir('save/')
    if not os.path.isdir('save/' + path_name):
        os.mkdir('save/' + path_name)
        print('create path save/' + path_name)
    filename = 'save/' + path_name + 'checkpoint'
    path = 'figs/' + path_name
    if not os.path.isdir('figs/'):
        os.mkdir('figs/')
    if not os.path.isdir(path):
        os.mkdir(path)
        print('create path ' + path)



    if train:
        print('training from scratch')
        old_var_list = init_variables(sess)
    #else:
    #    load_params(sess, filename, checkpoint)
    #   old_var_list = init_variables(sess, set(tf.trainable_variables()))
    checkpoint += 1

    # visualise the samples
    N_gen = 10 ** 2

    X_ph = tf.placeholder(tf.float32, shape=(batch_size, dimX), name='x_ph')

    # now start fitting
    N_task = len(labels)
    gen_ops = []
    X_valid_list = []
    X_test_list = []
    eval_func_list_ll = []
    eval_func_list_cla = []
    fits = []
    coresets = []
    cla = load_model(data_name)
    result_list = np.zeros((2,10,10))
    if method == 'onlinevi':
        shared_prior_params = init_shared_prior()
    n_layers_head = 2
    n_layers_enc = n_layers_shared + n_layers_head - 1
    for task in range(1, N_task + 1):
        # first load data
        if data_name == 'mnist':
            X_train, X_test, _, _ = load_mnist(digits=labels[task - 1], conv=False)
        if data_name == 'notmnist':
            X_train, X_test, _, _ = load_notmnist(data_path, digits=labels[task - 1], conv=False)
        N_train = int(X_train.shape[0] * 0.9) - coreset_size
        np.random.shuffle(X_train)
        X_valid_list.append(X_train[N_train+coreset_size:])
        X_coreset = X_train[:coreset_size]
        coresets.append(X_coreset)
        X_train = X_train[coreset_size:N_train+coreset_size]


        if degenerate_dataset:
            X_train = X_train[0,:].reshape(1,-1).repeat(X_train.shape[0], axis=0)

        X_test_list.append(X_test)

        # define the head net and the generator ops
        dec = generator(generator_head(dimZ, dimH, n_layers_head, 'gen_%d' % task), dec_shared)
        enc = encoder(dimX, dimH, dimZ, n_layers_enc, 'enc_%d' % task)
        gen_ops.append(construct_gen(dec, dimZ, sampling=False)(N_gen))
        print('construct eval function...')
        eval_func_list_ll.append(construct_eval_func(X_ph, enc, dec, ll, \
                                                  batch_size_ph, K=100, sample_W=False))
        eval_func_list_cla.append(construct_eval_func2(dec, cla, batch_size_ph, dimZ, task-1, sample_W=False))

        if not train:
            load_params(sess, filename, task - 1)
            old_var_list = init_variables(sess, set(tf.trainable_variables()))

        # then construct loss func and fit func
        print('construct fit function...')
        if train:
            if method == 'onlinevi':
                fit = construct_optimizer(X_ph, enc, dec, ll, X_train.shape[0], batch_size_ph, \
                                          shared_prior_params, task, K_mc)
        elif coreset_session:
            shared_prior_params = update_shared_prior(sess, shared_prior_params)
            private_prior_params = update_headnet_prior(sess, task)
            fit = construct_optimizer(X_ph, enc, dec, ll, X_coreset.shape[0], batch_size_ph, \
                                      shared_prior_params, task, K_mc,private_prior_params)
            fits.append(fit)
        # initialsise all the uninitialised stuff
        old_var_list = init_variables(sess, old_var_list)

        if train:
            # start training for each task
            fit(sess, X_train, n_iter, lr)
        elif coreset_session:
            for t in range(task):
                print("Coreset training for ", t)
                #fits[t](sess, coresets[t], n_coreset_iter, lr)
                print('task %d' % (t + 1), end=' ')
                result_list[0, task - 1, t], _ = eval_func_list_ll[t](sess, X_valid_list[t])

                print('task %d' % (t + 1), end=' ')
                result_list[1, task - 1, t], _ = eval_func_list_cla[t](sess)
                load_params(sess, filename, task - 1)


        if print_weights:
            #Print weight statistics
            print('SharedDec after task ', task, ":")
            for ind, l in enumerate(dec_shared.layers):
                muWmean, muWvar = [str(int(x.eval(session=sess)*100)/100) for x in tf.nn.moments(l.mu_W.value(), axes=[0, 1])]
                muBmean, muBvar = [str(int(x.eval(session=sess) * 100) / 100) for x in
                                   tf.nn.moments(l.mu_b.value(), axes=[0])]
                sigWmean, sigWvar = [str(int(x.eval(session=sess) * 100) / 100) for x in
                                   tf.nn.moments(l.log_sig_W.value(), axes=[0, 1])]
                sigBmean, sigBvar = [str(int(x.eval(session=sess) * 100) / 100) for x in
                                   tf.nn.moments(l.log_sig_b.value(), axes=[0])]
                print("Layer ",str(ind),": W = "+muWmean+"("+muWvar+")+-",sigWmean,"("+sigWvar+")  b="+muBmean+"("+muBvar+")+-",sigBmean,"("+sigBvar+")")


        # plot samples
        x_gen_list = sess.run(gen_ops, feed_dict={batch_size_ph: N_gen})
        for i in range(len(x_gen_list)):
            plot_images(x_gen_list[i], shape_high, path, \
                        data_name + '_gen_task%d_%d' % (task, i + 1))

        x_list = [x_gen_list[i][:1] for i in range(len(x_gen_list))]
        x_list = np.concatenate(x_list, 0)
        tmp = np.zeros([10, dimX])
        tmp[:task] = x_list
        if task == 1:
            x_gen_all = tmp
        else:
            x_gen_all = np.concatenate([x_gen_all, tmp], 0)

        # print test-ll on all tasks
        tmp_list = []
        if not coreset_session:
            for i in range(len(eval_func_list_ll)):
                print('task %d' % (i + 1), end=' ')
                result_list[0,task -1,i],_ = eval_func_list_ll[i](sess, X_valid_list[i])

            for i in range(len(eval_func_list_cla)):
                print('task %d' % (i + 1), end=' ')
                result_list[1,task -1,i],_ = eval_func_list_cla[i](sess)


        # save param values
        if train:
            save_params(sess, filename, checkpoint)
        checkpoint += 1

        # update regularisers/priors
        if method == 'onlinevi':
            # update prior
            print('update prior...')
            shared_prior_params = update_shared_prior(sess, shared_prior_params)
            # reset the variance of q
            update_q_sigma(sess)

        print()

    plot_images(x_gen_all, shape_high, path, data_name + '_gen_all')

    for i in range(len(result_list)):
        print(result_list[i])

    # save results
    fname = 'results/' + data_name + '_TF'
    np.save(fname, result_list)
    print('test-ll and test-cla results saved in', fname)


if __name__ == '__main__':
    data_name = 'mnist'
    methods = ['onlinevi']
    for method in methods:
        assert method in ['noreg', 'laplace', 'ewc', 'si', 'onlinevi']
        lbd = 1.0  # some placeholder, doesn't matter
        if method == 'si':
            lbd = 0.1
        if method == 'ewc':
            lbd = 10000.0
        if method == 'laplace':
            lbd = 1.0
        main(data_name, method, dimZ, dimH, n_channel, batch_size, K_mc, checkpoint, lbd)
