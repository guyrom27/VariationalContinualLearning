import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def KL_div_gaussian(mu_p, log_sig_p, mu_q, log_sig_q):
    # compute KL[p||q]
    precision_q = torch.exp(-2 * log_sig_q)
    kl = 0.5 * (mu_p - mu_q) ** 2 * precision_q - 0.5
    kl += log_sig_q - log_sig_p
    kl += 0.5 * torch.exp(2 * log_sig_p - 2 * log_sig_q)
    return torch.sum(kl, dim=list(range(1, len(kl.shape))))


# def KL_div_gaussian(mu_q, log_sig_q, mu_p, log_sig_p):
#    """
#    KL(q||p), gets log of sigma rather than sigma
#    """
#    kl = log_sig_p - log_sig_q + (0.5) * torch.exp(-2 * log_sig_p) * (
#            torch.exp(log_sig_q) ** 2 + (mu_q - mu_p) ** 2) - 1 / 2
#    return torch.sum(kl, dim=list(range(1, len(kl.shape))))


# In[99]:


def KL_div_gaussian_from_standard_normal(mu_q, log_sig_q):
    # 0,0 corresponds to N(0,1) due to the log_sig representation, works for multidim normal as well.
    return KL_div_gaussian(mu_q, log_sig_q, torch.zeros(1, device=device), torch.zeros(1, device=device))


# In[100]:


def Zs_to_mu_sig(Zs_params):
    dimZ = Zs_params.shape[1] // 2  # 1st is batch size 2nd is 2*dimZ
    mu_qz = Zs_params[:, :dimZ]
    log_sig_qz = Zs_params[:, dimZ:]
    return mu_qz, log_sig_qz


# In[101]:


forced_interval = (1e-9, 1.0)


def log_bernoulli(X, Mu_Reconstructed_X):
    """
    Mu_Reconstructed_X is the output of the decoder. We accept fractions, and project them to the interval 'forced_interval' for numerical stability
    """
    logprob = X * torch.log(torch.clamp(Mu_Reconstructed_X, *forced_interval)) \
              + (1 - X) * torch.log(torch.clamp((1.0 - Mu_Reconstructed_X), *forced_interval))

    return torch.sum(logprob.view(logprob.size()[0], -1), dim=1)  # sum all but first dim


def log_gaussian_prob(x, mu=torch.zeros(1, device=device), log_sig=torch.zeros(1, device=device)):
    logprob = -(0.5 * np.log(2 * np.pi) + log_sig) \
              - 0.5 * ((x - mu) / torch.exp(log_sig)) ** 2
    return torch.sum(logprob.view(logprob.size()[0], -1), dim=1)  # sum all but first dim


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
        logp += log_bernoulli(Xs, Mu_Ys) / NumLogPSamples
    return logp, kl_z

