#!/usr/bin/env python
# coding: utf-8

# In[92]:


import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

torch.manual_seed(123)
weight_print = False
data_print = False
loss_print = True


class mlp_layer(nn.Module):
    def __init__(self, d_in, d_out, activation):
        """
        Activation is a function (eg. torch.nn.functional.sigmoid/relu)
        """
        super().__init__()
        self.mu = nn.Linear(d_in, d_out)
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


from torch.distributions import Normal


class bayesian_mlp_layer(mlp_layer):
    def __init__(self, d_in, d_out, activation):
        """
        Activation is a function (eg. torch.nn.functional.sigmoid/relu)
        """
        super().__init__(d_in, d_out, activation)
        self.log_sigma = nn.Linear(d_in, d_out)
        self._init_log_sigma()
        # mu is initialized the same as non-Bayesian mlp

        """
        Attribute for now, but planning to do only "in-place" changes 
        """
        self.weight_sampler = Normal(self.mu.weight, torch.exp(self.log_sigma.weight))
        self.bias_sampler = Normal(self.mu.bias, torch.exp(self.log_sigma.bias))

    def forward(self, x, sampling=True):
        if sampling:
            my_lin = nn.Linear(*self.mu.weight.shape)
            my_lin.weight = nn.Parameter(self.weight_sampler.sample())
            my_lin.bias = nn.Parameter(self.bias_sampler.sample())

            # ADD other printers for log_sig and samples
            if (weight_print):
                print("Weights of mu DEC ", self.mu.weight)
                print("bias of mu DEC ", self.mu.bias)
                print("Weights of log_sig DEC ", self.log_sigma.weight)
                print("bias of log_sig DEC ", self.log_sigma.bias)

            return self.activation(my_lin(x))
        else:
            return super().forward(x)

    def _init_log_sigma(self):
        nn.init.constant_(self.log_sigma.weight, -6.0)
        nn.init.constant_(self.log_sigma.bias, -6.0)

    def get_posterior(self):
        return [(self.mu.weight, self.log_sigma.weight), (self.mu.bias, self.log_sigma.bias)]


# In[94]:


class NormalSamplingLayer:
    def __init__(self, d_out):
        self.d_out = d_out

    def __call__(self, mu_log_sigma_vec):
        return Normal(mu_log_sigma_vec[:, :self.d_out], torch.exp(mu_log_sigma_vec[:, self.d_out:])).sample()


# In[95]:


import itertools


class FunctionComposition:
    def __init__(self, f_list):
        assert (len(f_list) > 0)
        # for i in range(len(f_list)-1):
        # assert(f_list[i].d_out == f_list[i+1].d_in)
        self.f_list = f_list

    def __call__(self, x, *optional):
        for f in self.f_list:
            x = f(x, *optional)
        return x

    def parameters(self):
        return list(itertools.chain(*list(map(lambda f: f.parameters(), self.f_list))))

    @property
    def d_in(self):
        return self.f_list[0].d_in

    @property
    def d_out(self):
        return self.f_list[-1].d_out


# In[96]:


import itertools


class BayesianNet(FunctionComposition):
    def __init__(self, f_list):
        super().__init__(f_list)

    def get_posterior(self):
        return list(itertools.chain(*list(map(lambda f: f.get_posterior(), self.f_list))))


# In[97]:


class NNFactory:
    @classmethod
    def CreateNN(cls, dims, activations):
        assert (len(dims) - 1 == len(activations))
        layers = []
        for i in range(len(dims) - 1):
            layers.append(PrintLayer())
            layers.append(mlp_layer(dims[i], dims[i + 1], activations[i]))
        layers.append(PrintLayer())
        return FunctionComposition(layers)

    @classmethod
    def CreateBayesianNet(cls, dims, activations):
        assert (len(dims) - 1 == len(activations))
        layers = []
        for i in range(len(dims) - 1):
            layers.append(bayesian_mlp_layer(dims[i], dims[i + 1], activations[i]))
        return BayesianNet(layers)


# In[98]:

def KL_div_gaussian(mu_p, log_sig_p, mu_q, log_sig_q):
    # compute KL[p||q]
    precision_q = torch.exp(-2*log_sig_q)
    kl = 0.5 * (mu_p - mu_q)**2 * precision_q - 0.5
    kl += log_sig_q - log_sig_p
    kl += 0.5 * torch.exp(2 * log_sig_p - 2 * log_sig_q)
    #ind = list(range(1, len(mu_p.get_shape().as_list())))
    return torch.sum(kl,dim=list(range(1,len(kl.shape))))



#def KL_div_gaussian(mu_q, log_sig_q, mu_p, log_sig_p):
#    """
#    KL(q||p), gets log of sigma rather than sigma
#    """
#    return log_sig_p - log_sig_q + (0.5) * torch.exp(-2 * log_sig_p) * (
#            torch.exp(log_sig_q) ** 2 + (mu_q - mu_p) ** 2) - 1 / 2


# In[99]:


def KL_div_gaussian_from_standard_normal(mu_q, log_sig_q):
    # 0,0 corresponds to N(0,1) due to the log_sig representation, works for multidim normal as well.
    return KL_div_gaussian(mu_q, log_sig_q, torch.zeros(1), torch.zeros(1))


# In[100]:


def Zs_to_mu_sig(Zs_params):
    dimZ = Zs_params.shape[1] // 2  # 1st is batch size 2nd is 2*dimZ
    mu_qz = Zs_params[:, :dimZ]
    log_sig_qz = Zs_params[:, dimZ:]
    return mu_qz, log_sig_qz


# In[101]:


forced_interval = (1e-9, 1.0)


def log_bernouillli(X, Mu_Reconstructed_X):
    """
    Mu_Reconstructed_X is the output of the decoder. We accept fractions, and project them to the interval 'forced_interval' for numerical stability
    """
    logprob = X * torch.log(torch.clamp(Mu_Reconstructed_X, *forced_interval)) \
              + (1 - X) * torch.log(torch.clamp((1.0 - Mu_Reconstructed_X), *forced_interval))

    return torch.sum(logprob, dim=-1)


# In[102]:


def log_P_y_GIVEN_x(Xs, enc, sample_and_decode, NumLogPSamples=100):
    """
    Returns logP(Y|X), KL(Z||Normal(0,1))
    """
    Zs_params = enc(Xs)
    mu_qz, log_sig_qz = Zs_to_mu_sig(Zs_params)
    kl_z = KL_div_gaussian_from_standard_normal(mu_qz, log_sig_qz)
    logp = 0.0
    for _ in range(NumLogPSamples):
        # The Zs_params are the deterministic result of enc(Xs) so we don't recalculate them
        Mu_Ys = sample_and_decode(Zs_params)
        logp += log_bernouillli(Xs, Mu_Ys) / NumLogPSamples
    return logp, kl_z


# In[110]:


class SharedDecoder(nn.Module):
    def __init__(self, dims, activations):
        super().__init__()
        self.net = NNFactory.CreateBayesianNet(dims, activations)
        self._init_prior()

    def __call__(self, Xs):
        return self.net(Xs)

    def _init_prior(self):
        """
        Initialize a constant tensor that corresponds to a prior distribution over all the weights
        which is standard normal
        """
        self.prior = [(torch.zeros(mu.shape), torch.zeros(log_sig.shape)) for mu, log_sig in self.net.get_posterior()]

    def update_prior(self):
        """
        Copy the current posterior to a constant tensor, which will be used as prior for the next task
        """
        self.prior = [(mu.clone().detach(), log_sig.clone().detach()) for mu, log_sig in self.net.get_posterior()]

    def KL_from_prior(self):
        params = [(*post, *prior) for (post, prior) in zip(self.net.get_posterior(), self.prior)]
        KL = torch.zeros(1)
        for param in params:
            unsqueezed_param = list(map( lambda x: x.unsqueeze(0), param ))
            tmp = KL_div_gaussian(*unsqueezed_param)
            KL += tmp.squeeze()

        return KL.item()

    def parameters(self, **kwargs):
        return self.net.parameters()

    @property
    def d_in(self):
        return self.net.d_in

    @property
    def d_out(self):
        return self.net.d_out


# In[113]:


import torch.nn as nn
import torch.optim


# BayesianVAE
class TaskModel(nn.Module):
    def __init__(self, enc_dims_activations, dec_head_dims_activations, dec_shared, learning_rate=1e-4):
        super().__init__()
        self.enc = NNFactory.CreateNN(*enc_dims_activations)
        self.dec_head = NNFactory.CreateBayesianNet(*dec_head_dims_activations)
        self.dec_shared = dec_shared
        self.printer = PrintLayer()

        self.sampler = NormalSamplingLayer(self.dec_head.d_in)
        self.sample_and_decode = FunctionComposition(
            [self.sampler, self.dec_head, self.dec_shared])

        # update just before training
        self.DatasetSize = None

        # Guards from retraining- train only once
        self.TrainGuard = True

        self.optimizer = self._create_optimizer(learning_rate)

    def forward(self, Xs):
        logp, kl_z = log_P_y_GIVEN_x(Xs, self.enc, self.sample_and_decode)
        kl_shared_dec_Qt_2_PREV_Qt = self.dec_shared.KL_from_prior()

        if (loss_print):
            print("Log_like", torch.mean(logp))
            print("KL Z", torch.mean(kl_z))
            print("KL Qt vs prev Qt", (kl_shared_dec_Qt_2_PREV_Qt / self.DatasetSize))

        # We ignore the kl(private dec || Normal(0,1) ) like the authors did
        ELBO = torch.mean(logp) - torch.mean(kl_z) - (kl_shared_dec_Qt_2_PREV_Qt / self.DatasetSize)
        return -ELBO

    def parameters(self, **kwargs):
        return self.enc.parameters() + self.dec_shared.parameters() + self.dec_head.parameters()

    def _create_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def _update_prior(self):
        self.dec_shared.update_prior()
        # no other priors should be updated. they are trained once.
        return

    def train(self, n_epochs, task_trainloader):
        # We don't intend a TaskModel to be trained more than once
        assert (self.TrainGuard)
        self.TrainGuard = False

        self.DatasetSize = len(task_trainloader.dataset)

        # loop over the dataset multiple times
        for epoch in range(n_epochs):
            print("starting epoch " + str(epoch))
            running_loss = 0.0
            for i, data in enumerate(task_trainloader):
                # get the inputs
                inputs, labels = data

                # step
                self.optimizer.zero_grad()
                # loss = self(inputs.view(-1, self.enc.d_in))
                loss = self(inputs.view(-1, 28 ** 2))
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()  # ?
                if i % 1 == 0:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss))
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


import torch.utils.data
import torchvision


def create_mnist_single_digit_loaders(b_size=10, train_data=True):
    dataset = torchvision.datasets.MNIST(root='./data', train=train_data, download=True,
                                         transform=torchvision.transforms.ToTensor())

    for i in range(10):
        partial_dataset = torch.utils.data.Subset(dataset, torch.nonzero(dataset.train_labels == i).squeeze())

        # NOT Repeating the original "mistake"
        # train_idx = len(partial_trainset) * 0.9
        partial_loader = torch.utils.data.DataLoader(partial_dataset, batch_size=b_size, shuffle=True)
        yield partial_loader


# In[136]:


dimX = 28 * 28
dimH = 500
dimZ = 50
batch_size = 50
n_epochs = 1  # 200

# Shared decoder
dec_shared_dims = [dimH, dimH, dimX]
dec_shared_activations = [F.relu, torch.sigmoid]

# Encoder
enc_dims = [dimX, dimH, dimH, dimZ * 2]
enc_activations = [F.relu, F.relu, lambda x: x]

# Private decoder (Head)
dec_head_dims = [dimZ, dimH, dimH]
dec_head_activations = [F.relu, F.relu]


def main():
    dec_shared = SharedDecoder(dec_shared_dims, dec_shared_activations)

    # TODO: check dec_shared.parameters()

    # this can be any iterable
    task_loaders = list(create_mnist_single_digit_loaders(batch_size))

    models = []

    # this may train the classifier to generate test_classifier
    # evaluator = Evaluations()

    # A task corresponds to a digit
    for task_id, loader in enumerate(task_loaders):
        print("starting task " + str(task_id))
        task_model = TaskModel((enc_dims, enc_activations), (dec_head_dims, dec_head_activations), dec_shared)
        assert (len(task_model.parameters()) == 4 * (2 + 2) + 2 * 3)
        models.append(task_model)
        print("starting training")
        task_model.train(n_epochs, loader)
        # make sure you don't change the model params inside the eval
        # evaluator.create_task_evaluations(models)

    # evaluator.report_results()


# In[137]:


main()

# In[ ]:
