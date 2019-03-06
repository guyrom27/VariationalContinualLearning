import numpy as np
import sys
sys.path.extend(['alg/'])
import discriminative.vcl as vcl
import discriminative.coreset as coreset
from discriminative.DataGenerator import PermutedMnistGenerator



hidden_size = [100, 100]
batch_size = 256
no_epochs = 100
single_head = True
num_tasks = 10

np.random.seed(1)
run_vanilla = True
if run_vanilla:
    # Run vanilla VCL
    coreset_size = 0
    data_gen = PermutedMnistGenerator(num_tasks)
    vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
        coreset.rand_from_batch, coreset_size, batch_size, single_head)
    np.save("./results/VCL{}".format(""), vcl_result)
    print(vcl_result)

# Run random coreset VCL
np.random.seed(1)

for coreset_size in [200,400,1000,2500,5000]:
    data_gen = PermutedMnistGenerator(num_tasks)
    rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
        coreset.rand_from_batch, coreset_size, batch_size, single_head)
    np.save("./results/rand-VCL-{}".format(coreset_size), rand_vcl_result)
    print(rand_vcl_result)


# Run k-center coreset VCL
np.random.seed(1)
coreset_size = 200
data_gen = PermutedMnistGenerator(num_tasks)
kcen_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
    coreset.k_center, coreset_size, batch_size, single_head)
print(kcen_vcl_result)
np.save("./results/pca-kcen-VCL-{}".format(coreset_size), kcen_vcl_result)

# Plot average accuracy
vcl_avg = np.nanmean(vcl_result, 1)
rand_vcl_avg = np.nanmean(rand_vcl_result, 1)
kcen_vcl_avg = np.nanmean(kcen_vcl_result, 1)
