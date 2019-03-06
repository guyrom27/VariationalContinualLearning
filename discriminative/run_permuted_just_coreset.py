import numpy as np
import tensorflow as tf
import gzip
import pickle as cp
import sys
sys.path.extend(['alg/'])
import discriminative.vcl as vcl
import discriminative.coreset as coreset
import discriminative.utils as utils
from discriminative.DataGenerator import PermutedMnistGenerator

hidden_size = [100, 100]
batch_size = 256
no_epochs = 100
single_head = True
num_tasks = 10

np.random.seed(0)


for coreset_size in [400,1000,2500,5000]:
    data_gen = PermutedMnistGenerator(num_tasks)
    vcl_result = vcl.run_coreset_only(hidden_size, no_epochs, data_gen,
        coreset.rand_from_batch, coreset_size, batch_size, single_head)
    np.save("./results/only-coreset-{}".format(coreset_size), vcl_result)
    print(vcl_result)


