import time
import torch
import math_utils
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def IS_estimate(x, task_model, K):
    x = x.view(-1, 28 ** 2)
    x_rep = x.repeat([K, 1]).to(device=device)
    assert(x_rep.size()[0] < 6000)

    N = x.size()[0]
    Zs_params = task_model.enc(x_rep)
    mu_qz, log_sig_qz = math_utils.Zs_to_mu_sig(Zs_params)
    z = task_model.sampler(Zs_params)
    mu_x = task_model.dec_shared(task_model.dec_head(z))
    logp = math_utils.log_bernoulli(x_rep, mu_x)

    log_prior = math_utils.log_gaussian_prob(z)
    logq = math_utils.log_gaussian_prob(z, mu_qz, log_sig_qz)
    kl_z = logq - log_prior

    bound = torch.reshape(logp - kl_z, (K, N))
    bound_max = torch.max(bound, 0)[0]
    bound -= bound_max
    log_norm = torch.log(torch.clamp(torch.mean(torch.exp(bound), 0), 1e-9, np.inf))

    test_ll = log_norm + bound_max
    test_ll_mean = torch.mean(test_ll).item()
    test_ll_var = torch.mean((test_ll - test_ll_mean) ** 2).item()

    return test_ll_mean, test_ll_var


class Evaluation:

    def __init__(self, should_print=True, K=100):
        self.should_print = should_print
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
        if self.should_print:
            print("test_ll=%.2f, ste=%.2f, time=%.2f" \
                  % (bound_tot, np.sqrt(bound_var / N), end - begin))
        return bound_tot, np.sqrt(bound_var / N)

