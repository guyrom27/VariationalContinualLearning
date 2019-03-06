import numpy as np
import gzip
import pickle as cp
import sys
sys.path.extend(['alg/'])
import discriminative.vcl as vcl
import discriminative.coreset as coreset
import discriminative.utils as utils
from discriminative.DataGenerator import SplitMnistGenerator
from copy import deepcopy


hidden_size = [256, 256]
batch_size = None
no_epochs = 120
single_head = False
run_coreset_only = False
# Run vanilla VCL

if run_coreset_only:
    np.random.seed(0)
    coreset_size = 40
    #data_gen = SplitMnistGenerator()
    #rand_vcl_result = vcl.run_vcl_vanilla(hidden_size, no_epochs, data_gen,
    #    coreset.rand_from_batch, coreset_size, batch_size, single_head)
    #print(rand_vcl_result)
    #np.save("./results/rand-coreset-only-split{}".format(""), rand_vcl_result)
    # Run k-center coreset VCL
    np.random.seed(1)

    data_gen = SplitMnistGenerator()
    kcen_vcl_result = vcl.run_vcl_vanilla(hidden_size, no_epochs, data_gen,
        coreset.k_center, coreset_size, batch_size, single_head)
    print(kcen_vcl_result)
    np.save("./results/kcen-coreset-only-split{}".format(""), kcen_vcl_result)

else:
    #coreset_size = 0
    #data_gen = SplitMnistGenerator()
    #vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
    #    coreset.rand_from_batch, coreset_size, batch_size, single_head)
    #print(vcl_result)
    #np.save("./results/VCL-split{}".format(""), vcl_result)
    # Run random coreset VCL
    #np.random.seed(0)
    coreset_size = 40
    #data_gen = SplitMnistGenerator()
    #rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
    #    coreset.rand_from_batch, coreset_size, batch_size, single_head)
    #print(rand_vcl_result)
    #np.save("./results/randVCL-split{}".format(""), rand_vcl_result)

    # Run k-center coreset VCL
    np.random.seed(1)

    data_gen = SplitMnistGenerator()
    kcen_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
        coreset.k_center, coreset_size, batch_size, single_head)
    print(kcen_vcl_result)
    np.save("./results/kcenVCL-split{}".format(""), kcen_vcl_result)
    # Plot average accuracy
    vcl_avg = np.nanmean(vcl_result, 1)
    rand_vcl_avg = np.nanmean(rand_vcl_result, 1)
    kcen_vcl_avg = np.nanmean(kcen_vcl_result, 1)
    utils.plot('results/split.jpg', vcl_avg, rand_vcl_avg, kcen_vcl_avg)
