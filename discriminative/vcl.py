import numpy as np
import tensorflow as tf
import utils
from multihead_models import Vanilla_NN, MFVI_NN
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_vcl(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []

    all_acc = np.array([])

    for task_id in range(data_gen.max_iter):
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        # Set the readout head to train
        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        # Train network with maximum likelihood to initialize first model
        if task_id == 0:
            ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0])
            ml_model.train(x_train, y_train, task_id, no_epochs, bsize)
            mf_weights = ml_model.get_weights()
            mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0], single_head, prev_means=mf_weights)


        # Select coreset if needed
        if coreset_size > 0:
            x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)

        # Train on non-coreset data
        mf_model.train(x_train, y_train, head, no_epochs, bsize)
        mf_model.save_weights()

        # Incorporate coreset data and make prediction
        #acc = utils.get_scores(mf_model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, no_epochs, single_head, batch_size)
        #all_acc = utils.concatenate_results(acc, all_acc)

        mf_model.load_weights()

        ##Create head + update prior
        mf_model.new_task()

    return all_acc
