import time
import torch
import numpy as np


def Zs_to_mu_sig(Zs_params):
    dimZ = Zs_params.shape[1] // 2  # 1st is batch size 2nd is 2*dimZ
    mu_qz = Zs_params[:, :dimZ]
    log_sig_qz = Zs_params[:, dimZ:]
    return mu_qz, log_sig_qz


def log_gaussian_prob(x, mu=0.0, log_sig=0.0):
    logprob = -(0.5 * np.log(2 * np.pi) + log_sig) \
              - 0.5 * ((x - mu) / torch.exp(log_sig)) ** 2
    ind = list(range(1, len(x.size())))
    return torch.sum(logprob, ind)


def log_bernoulli_prob(x, p=0.5):
    logprob = x * torch.log(torch.clamp(p, 1e-9, 1.0)) \
              + (1 - x) * torch.log(torch.clamp(1.0 - p, 1e-9, 1.0))
    ind = list(range(1, len(x.size())))
    return torch.sum(logprob, ind)


def IS_estimate(x, task_model, K):
    x = x.view(-1, 28 ** 2)
    if len(x.size()) == 2:
        x = x.repeat([K, 1])
    assert(x.size()[0] < 6000)

    N = x.size()[0]
    z_params = task_model.enc(x)
    mu_qz, log_sig_qz = Zs_to_mu_sig(z_params)
    z = task_model.sampler(z_params)
    mu_x = task_model.sample_and_decode(z_params)

    log_prior = log_gaussian_prob(z, log_sig=torch.zeros(z.size()))
    logq = log_gaussian_prob(z, mu_qz, log_sig_qz)
    kl_z = logq - log_prior
    logp = log_bernoulli_prob(x, mu_x)

    bound = torch.reshape(logp - kl_z, (1, N))
    bound_max = torch.max(bound, 0)[0]
    bound -= bound_max
    log_norm = torch.log(torch.clamp(torch.mean(torch.exp(bound), 0), 1e-9, np.inf))

    test_ll = log_norm + bound_max
    test_ll_mean = torch.mean(test_ll).item()
    test_ll_var = torch.mean((test_ll - test_ll_mean) ** 2).item()

    return test_ll_mean, test_ll_var


class Evaluation:

    def __init__(self, K=100):
        self.K = K

    def __call__(self, task_id, task_model, loader):
        N = 0
        n_iter_vae = len(loader)
        bound_tot = 0.0
        bound_var = 0.0
        begin = time.time()
        for j, data in enumerate(loader):
            inputs, labels = data
            N += len(inputs)
            logp_mean, logp_var = IS_estimate(inputs, task_model, self.K)

            bound_tot += logp_mean / n_iter_vae
            bound_var += logp_var / n_iter_vae
        end = time.time()
        print("test_ll=%.2f, ste=%.2f, time=%.2f" \
                  % (bound_tot, np.sqrt(bound_var / N), end - begin))
        return bound_tot, np.sqrt(bound_var / N)

