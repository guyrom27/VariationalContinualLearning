#!/usr/bin/env python
# coding: utf-8

# In[92]:

import torch
torch.manual_seed(123)


output_path = None

if (not output_path is None):
    import sys
    sys.stdout = open(output_path, mode='w')

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.backends.cudnn.benchmark = True



import itertools
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim

import torch.utils.data
import torchvision

import math_utils
import evaluate_YvsX_log_like
import EvaluateClassifierUncertainty




Train = True
max_task = 10

weight_print = False
data_print = False
loss_print = False


scale_down_090 = True
degenerate_dataset = False


dimX = 28 * 28
dimH = 500
dimZ = 50
batch_size = 50
n_epochs = 200

# Shared decoder
dec_shared_dims = [dimH, dimH, dimX]
dec_shared_activations = [F.relu, torch.sigmoid]

# Encoder
enc_dims = [dimX, dimH, dimH, dimH, dimZ * 2]
enc_activations = [F.relu, F.relu, F.relu, lambda x: x]

# Private decoder (Head)
dec_head_dims = [dimZ, dimH, dimH]
dec_head_activations = [F.relu, F.relu]


class mlp_layer(nn.Module):
    def __init__(self, d_in, d_out, activation):
        """
        Activation is a function (eg. torch.nn.functional.sigmoid/relu)
        """
        super().__init__()
        self.mu = nn.Linear(d_in, d_out).to(device=device)
        self._init_weights(d_in, d_out)
        self.activation = activation

    def forward(self, x):
        if (weight_print):
            print("Weights of ENC ", self.mu.weight)
            print("bias of ENC ", self.mu.bias)
        return self.activation(self.mu(x))

    def _init_weights(self, input_size, output_size, constant=1.0):
        scale = constant * np.sqrt(6.0 / (input_size + output_size))
        assert (output_size > 0)
        nn.init.uniform_(self.mu.weight, -scale, scale)
        nn.init.zeros_(self.mu.bias)

    @property
    def d_out(self):
        return self.mu.weight.shape[0]

    @property
    def d_in(self):
        return self.mu.weight.shape[1]


# In[93]:





class bayesian_mlp_layer(mlp_layer):
    def __init__(self, d_in, d_out, activation):
        """
        Activation is a function (eg. torch.nn.functional.sigmoid/relu)
        """
        super().__init__(d_in, d_out, activation)
        self.log_sigma = nn.Linear(d_in, d_out).to(device=device)
        self._init_log_sigma()
        # mu is initialized the same as non-Bayesian mlp

        self.w_standard_normal_sampler = Normal(torch.zeros(self.mu.weight.shape, device=device), torch.ones(self.mu.weight.shape, device=device))
        self.b_standard_normal_sampler = Normal(torch.zeros(self.mu.bias.shape, device=device), torch.ones(self.mu.bias.shape, device=device))

        self.sampling = True

    def forward(self, x):

        if (weight_print):
            print("Weights of mu DEC ", self.mu.weight)
            print("bias of mu DEC ", self.mu.bias)
            print("Weights of log_sig DEC ", self.log_sigma.weight)
            print("bias of log_sig DEC ", self.log_sigma.bias)

        if self.sampling:

            sampled_W = (self.mu.weight + self.w_standard_normal_sampler.sample().to(device=device) * torch.exp(self.log_sigma.weight))
            sampled_b = (self.mu.bias + self.b_standard_normal_sampler.sample().to(device=device) * torch.exp(self.log_sigma.bias))
            return self.activation(torch.einsum('ij,bj->bi',[sampled_W, x]) + sampled_b)
        else:
            return super().forward(x)

    def _init_log_sigma(self):
        nn.init.constant_(self.log_sigma.weight, -6.0)
        nn.init.constant_(self.log_sigma.bias, -6.0)

    def get_posterior(self):
        return [(self.mu.weight, self.log_sigma.weight), (self.mu.bias, self.log_sigma.bias)]


# In[94]:


class NormalSamplingLayer(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.d_out = d_out

    def forward(self, mu_log_sigma_vec):
        return Normal(mu_log_sigma_vec[:, :self.d_out], torch.exp(mu_log_sigma_vec[:, self.d_out:])).sample().to(device=device)



# In[98]:


# In[110]:



class SharedDecoder(nn.Module):
    def __init__(self, dims, activations):
        super().__init__()
        # Not sure if device does anything
        self.net = nn.Sequential(*[bayesian_mlp_layer(dims[i], dims[i + 1], activations[i]) \
                                   for i in range(len(activations))])
        self._init_prior()

    def forward(self, Xs):
        return self.net(Xs)

    def _get_posterior(self):
        return list(itertools.chain(*list(map(lambda f: f.get_posterior(), self.net.children()))))

    def _init_prior(self):
        """
        Initialize a constant tensor that corresponds to a prior distribution over all the weights
        which is standard normal
        """
        self.prior = [(torch.zeros(mu.shape, device=device), torch.zeros(log_sig.shape, device=device)) for mu, log_sig in self._get_posterior()]

    def update_prior(self):
        """
        Copy the current posterior to a constant tensor, which will be used as prior for the next task
        """
        posterior = self._get_posterior()
        self.prior = [(mu.clone().detach(), log_sig.clone().detach()) for mu, log_sig in posterior]
        #update the new posterior's log_sig to -6
        for mu_sig in posterior:
            mu_sig[1].fill_(-6.0)


    def KL_from_prior(self):
        params = [(*post, *prior) for (post, prior) in zip(self._get_posterior(), self.prior)]
        KL = torch.zeros(1, device=device).squeeze() #don't know how to generate a zero scalar
        for param in params:
            unsqueezed_param = list(map(lambda x: x.unsqueeze(0), param))
            tmp = math_utils.KL_div_gaussian(*unsqueezed_param)
            KL += tmp.squeeze()

        return KL

    @property
    def d_in(self):
        return self.net.d_in

    @property
    def d_out(self):
        return self.net.d_out


# In[113]:




# BayesianVAE
class TaskModel(nn.Module):
    def __init__(self, enc_dims_activations, dec_head_dims_activations, dec_shared, learning_rate=1e-4):
        super().__init__()
        my_enc_dims, my_enc_activations = enc_dims_activations
        # Not sure if device does anything
        self.enc = nn.Sequential(*[mlp_layer(my_enc_dims[i], my_enc_dims[i + 1], my_enc_activations[i])
                                   for i in range(len(my_enc_activations))])

        my_dec_head_dims, my_dec_head_activations = dec_head_dims_activations

        # Not sure if device does anything
        self.dec_head = nn.Sequential(
            *[bayesian_mlp_layer(my_dec_head_dims[i], my_dec_head_dims[i + 1], my_dec_head_activations[i])
              for i in range(len(my_dec_head_activations))])

        self.dec_shared = dec_shared
        self.printer = PrintLayer()

        # Not sure if device does anything
        self.sampler = NormalSamplingLayer(my_dec_head_dims[0])
        self.sample_and_decode = nn.Sequential(*[self.sampler, self.dec_head, self.dec_shared])
        self.decode = nn.Sequential(*[self.dec_head, self.dec_shared])

        # update just before training
        self.DatasetSize = None

        # Guards from retraining- train only once
        self.TrainGuard = True

        self.optimizer = self._create_optimizer(learning_rate)

    def set_sampling(self, sampling):
        for model in self.dec_head.children():
            model.sampling = sampling
        for model in self.dec_shared.net.children():
            model.sampling = sampling

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load_model(cls, path, uninitialized_instance):
        uninitialized_instance.load_state_dict(torch.load(path,map_location=device))
        uninitialized_instance.eval()
        return uninitialized_instance

    def forward(self, Xs):
        logp, kl_z = math_utils.log_P_y_GIVEN_x(Xs, self.enc, self.sample_and_decode)
        kl_shared_dec_Qt_2_PREV_Qt = self.dec_shared.KL_from_prior()

        if loss_print:
            print("Log_like", "\tKL Z","\tKL Qt vs prev Qt")
            print(int(torch.mean(logp).item()),'\t',int(torch.mean(kl_z).item()), '\t', int(kl_shared_dec_Qt_2_PREV_Qt / self.DatasetSize))

        # We ignore the kl(private dec || Normal(0,1) ) like the authors did
        ELBO = torch.mean(logp) - torch.mean(kl_z) - (kl_shared_dec_Qt_2_PREV_Qt / self.DatasetSize)
        return -ELBO

    def _create_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def _update_prior(self):
        self.dec_shared.update_prior()
        # no other priors should be updated. they are trained once.
        return

    def train_model(self, n_epochs, task_trainloader, DatasetSize):
        # We don't intend a TaskModel to be trained more than once
        assert (self.TrainGuard)
        self.TrainGuard = False

        self.DatasetSize = DatasetSize

        # loop over the dataset multiple times
        for epoch in range(n_epochs):
            print("starting epoch " + str(epoch))
            running_loss = 0.0
            for i, data in enumerate(task_trainloader):
                global loss_print
                loss_print = (i % 20 == 19)
                # get the inputs
                inputs, labels = data
                #Migrate to device (gpu if possible)
                inputs = inputs.to(device=device)
                # step
                self.optimizer.zero_grad()
                # loss = self(inputs.view(-1, self.enc.d_in))
                loss = self(inputs.view(-1, 28 ** 2))
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()  # ?
                if i % 20 == 19:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss/20))
                    running_loss = 0.0
        # This will set the prior to the current posterior, before we start to change it during training
        self._update_prior()
        self.DatasetSize = None


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
        self.count = 0

    def forward(self, x):
        if (data_print):
            self.count += 1
            print("PRINTER LAYER")
            print(self.count)
            print(x[0].shape)
            print(x[0])
        return x


# In[135]:

def single_digit_loader(X, label, b_size=10):
    X.reshape(-1,dimX)
    N = X.shape[0]
    for i in range(N//b_size):
        yield (torch.from_numpy(X[i*b_size:(i+1)*b_size,:]), torch.from_numpy(np.ones(b_size,dtype=int)))
    if (N/b != 0.0):
        end = list(range((N//b_size)*b_size,X.shape[0]))
        n_missing = b_size - len(end)
        last_batch_ind = end + list(range(n_missing))
        yield (torch.from_numpy(X[last_batch_ind,:]), torch.from_numpy(np.ones(b_size,dtype=int)*label))



def create_mnist_single_digit_loaders(b_size=10, train_data=True):
    import generative.models.mnist
    for i in range(10):
        X_train, X_test, Y_train, Y_test = generative.models.mnist.load_mnist(digits = [i])
        if degenerate_dataset:
            X_train = X_train[0,:].reshape(1,-1).repeat(X_train.shape[0], axis=0)
        if train_data:
            N_train = int(X_train.shape[0] * 0.9) if scale_down_090 else X_train.shape[0]
            X_train = X_train[:N_train]
            yield (X_train.shape[0], single_digit_loader(X_train, i, b_size))
        else:
            yield (X_test.shape[0], single_digit_loader(X_test, i, b_size))





#the original code
"""def create_mnist_single_digit_loaders(b_size=10, train_data=True):
    dataset = torchvision.datasets.MNIST(root='./data',    train=train_data, download=True,
                                         transform=torchvision.transforms.ToTensor())

    for i in range(10):
        partial_dataset = torch.utils.data.Subset(dataset, torch.nonzero(dataset.targets == i).squeeze())

        # NOT Repeating the original "mistake"
        # train_idx = len(partial_trainset) * 0.9
        partial_loader = torch.utils.data.DataLoader(partial_dataset, batch_size=b_size, shuffle=True)
        yield (len(partial_dataset), partial_loader)
"""

# In[136]:


def path(after, i):
    return './checkpoint_params/after_task_' + str(after) + '_params_for_task_' + str(i) + '.pt'

def load_models(after):
    dec_shared = SharedDecoder(dec_shared_dims, dec_shared_activations)
    models = []
    for task_id in range(after+1): #range is 0 based
        # Not sure if device does anything
        task_model = TaskModel((enc_dims, enc_activations), (dec_head_dims, dec_head_activations), dec_shared)
        models.append(TaskModel.load_model(path(after,task_id), task_model))
    return models

import generative.models.visualisation
def generate_pictures(task_models, n_pics=100):
    with torch.no_grad():
        for task_id, task_model in enumerate(task_models):
            pics = task_model.sample_and_decode(torch.ones(n_pics, dimZ * 2, device=device))
            pics = pics.cpu()
            generative.models.visualisation.plot_images(pics, (28, 28), './figs/', 'after_task_'+str(len(task_models))+'_task_'+str(task_id))


def main():
    dec_shared = SharedDecoder(dec_shared_dims, dec_shared_activations)

    task_loaders = zip(create_mnist_single_digit_loaders(batch_size), create_mnist_single_digit_loaders(batch_size, train_data=False))

    test_loaders = []
    models = []

    evaluators = [evaluate_YvsX_log_like.Evaluation(),
                  EvaluateClassifierUncertainty.EvaluateClassifierUncertainty('./classifier_params')] #classifier is loaded. asssumes already trained

    # A task corresponds to a digit
    for task_id, ((N_train, train_loader),(N_test, test_loader)) in enumerate(task_loaders):
        if task_id>max_task:
            break
        print("starting task " + str(task_id))
        if (Train):
            task_model = TaskModel((enc_dims, enc_activations), (dec_head_dims, dec_head_activations), dec_shared)
            models.append(task_model)
            task_model.train_model(n_epochs, train_loader, N_train)
        else:
            models = load_models(task_id)
            task_model = models[-1]
        test_loaders.append(test_loader)
        #Disable gradient calculation during evaluation
        with torch.no_grad():
            if (Train):
                for i, model in enumerate(models):
                    model.save_model(path(task_id, i))
            generate_pictures(models)
            for test_task_id, loader in enumerate(test_loaders):
                for evaluator in evaluators:
                    evaluator(test_task_id, models[test_task_id], loader)

        print()

# In[137]:

main()

# In[ ]:
