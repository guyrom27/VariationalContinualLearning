import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import truncnorm
from torch.autograd import Variable
from copy import deepcopy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(0)
#tf.set_random_seed(0)

# variable initialization functions
def truncated_normal(size, stddev=1, variable = False, mean=0):
    mu, sigma = mean, stddev
    lower, upper= -2 * sigma, 2 * sigma
    X = truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    X_tensor = torch.Tensor(data = X.rvs(size)).to(device = device)
    X_tensor.requires_grad = variable
    return X_tensor

def init_tensor(value,  dout, din = 1, variable = False):
    if din != 1:
        x = value * torch.ones([din, dout]).to(device = device)
    else:
        x = value * torch.ones([dout]).to(device = device)
    x.requires_grad=variable

    return x

class Cla_NN(object):
    def __init__(self, input_size, hidden_size, output_size, training_size):
        return


    def train(self, x_train, y_train, task_idx, no_epochs=1000, batch_size=100, display_epoch=5):
        N = x_train.shape[0]
        if batch_size > N:
            batch_size = N

        costs = []
        # Training cycle
        for epoch in range(no_epochs):
            perm_inds = np.arange(x_train.shape[0])
            np.random.shuffle(perm_inds)
            cur_x_train = x_train[perm_inds]
            cur_y_train = y_train[perm_inds]

            avg_cost = 0.
            total_batch = int(np.ceil(N * 1.0 / batch_size))
            # Loop over all batches
            for i in range(total_batch):
                start_ind = i*batch_size
                end_ind = np.min([(i+1)*batch_size, N])
                batch_x = torch.Tensor(cur_x_train[start_ind:end_ind, :]).to(device = device)
                batch_y = torch.Tensor(cur_y_train[start_ind:end_ind]).to(device = device)


                self.optimizer.zero_grad()
                cost = self.get_loss(batch_x, batch_y, task_idx)
                cost.backward()
                self.optimizer.step()

                # Compute average loss
                avg_cost += cost / total_batch
            # Display logs per epoch step
            if epoch % display_epoch == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
            costs.append(avg_cost)
        print("Optimization Finished!")
        return costs

    def prediction_prob(self, x_test, task_idx):
        prob = torch.nn.softmax(self._prediction(x_test, task_idx, self.no_pred_samples))
        return prob


    def get_weights(self):
        return self.weights



""" Neural Network Model """
class Vanilla_NN(Cla_NN):
    def __init__(self, input_size, hidden_size, output_size, training_size, learning_rate=0.001):
        #
        super(Vanilla_NN, self).__init__(input_size, hidden_size, output_size, training_size)
        # # init weights and biases
        self.W, self.b, self.W_last, self.b_last, self.size = self.create_weights(
                 input_size, hidden_size, output_size)
        self.no_layers = len(hidden_size) + 1
        #self.pred = self._prediction(self.x, self.task_idx)
        #self.cost = - self._logpred(self.x, self.y, self.task_idx)
        self.weights = self.W + self.b + self.W_last + self.b_last
        self.training_size = training_size
        self.optimizer = optim.SGD(self.weights, lr=learning_rate)

    def _prediction(self, inputs, task_idx):
        act = inputs
        for i in range(self.no_layers-1):
             pre = torch.add(torch.matmul(act, self.W[i]), self.b[i])
             act = F.relu(pre)
        pre = torch.add(torch.matmul(act, self.W_last[task_idx]), self.b_last[task_idx])
        return pre

    def _logpred(self, inputs, targets, task_idx):

        loss = torch.nn.CrossEntropyLoss()
        pred = self._prediction(inputs, task_idx)
        log_lik = - loss(F.softmax(pred), targets.type(torch.long))
        return log_lik

    def get_loss(self, batch_x, batch_y, task_idx):
        return self._logpred(batch_x, batch_y, task_idx)

    def create_weights(self, in_dim, hidden_size, out_dim):
        hidden_size = deepcopy(hidden_size)
        hidden_size.append(out_dim)
        hidden_size.insert(0, in_dim)

        no_layers = len(hidden_size) - 1
        W = []
        b = []
        W_last = []
        b_last = []
        for i in range(no_layers-1):
            din = hidden_size[i]
            dout = hidden_size[i+1]

            #Initializiation values of means
            W_m = truncated_normal([din, dout], stddev=0.1, variable = True)
            bi_m = truncated_normal([dout], stddev=0.1, variable = True)

            #Append to list weights
            W.append(W_m)
            b.append(bi_m)

        Wi = truncated_normal([hidden_size[-2], out_dim], stddev=0.1, variable = True)
        bi = truncated_normal([out_dim], stddev=0.1, variable = True)
        W_last.append(Wi)
        b_last.append(bi)
        return W, b, W_last, b_last, hidden_size

""" Bayesian Neural Network with Mean field VI approximation """
class MFVI_NN(Cla_NN):
    def __init__(self, input_size, hidden_size, output_size, training_size,
        no_train_samples=10, no_pred_samples=100, prev_means=None, prev_log_variances=None, learning_rate=0.001,
        prior_mean=0, prior_var=1):

        super(MFVI_NN, self).__init__(input_size, hidden_size, output_size, training_size)

        m1, v1 = self.create_weights(
             input_size, hidden_size, output_size)

        self.input_size = input_size
        self.out_size = output_size
        self.size = deepcopy(hidden_size)

        hidden_size.append(self.out_size)
        hidden_size.insert(0, self.input_size)
        self.hidden_size_with_input_output = hidden_size

        self.W_m, self.b_m = m1[0], m1[1]
        self.W_v, self.b_v = v1[0], v1[1]

        self.W_last_m, self.b_last_m = [], []
        self.W_last_v, self.b_last_v = [], []

        m2, v2 = self.create_prior(input_size, self.hidden_size_with_input_output, output_size)


        self.prior_W_m, self.prior_b_m, = m2[0], m2[1]
        self.prior_W_v, self.prior_b_v = v2[0], v2[1]

        self.prior_W_last_m, self.prior_b_last_m = [], []
        self.prior_W_last_v, self.prior_b_last_v = [], []


        self.create_head()

        ##append the last layers to the general weights to keep track of the gradient easily
        m1.append(self.W_last_m)
        m1.append(self.b_last_m)
        v1.append(self.W_last_v)
        v1.append(self.b_last_v)

        r1 = m1 + v1
        self.weights = [item for sublist in r1 for item in sublist]

        self.no_layers = len(self.size) - 1
        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples
        self.training_size = training_size
        self.optimizer = optim.SGD(self.weights, lr=0.01)

    def get_loss(self, batch_x, batch_y, task_idx):
        return torch.div(self._KL_term(), self.training_size) - self._logpred(batch_x, batch_y, task_idx)

    def _prediction(self, inputs, task_idx, no_samples):
        return self._prediction_layer(inputs, task_idx, no_samples)

    # this samples a layer at a time
    def _prediction_layer(self, inputs, task_idx, no_samples):
        K = no_samples
        size = self.hidden_size_with_input_output


        act = torch.unsqueeze(inputs, 0).repeat([K, 1, 1])
        for i in range(self.no_layers-1):
            ##TODO: check dimensions
            din = self.hidden_size_with_input_output[i]
            dout = self.hidden_size_with_input_output[i+1]
            eps_w = torch.normal(torch.zeros((K, din, dout)), torch.ones((K, din, dout))).to(device = device)
            eps_b = torch.normal(torch.zeros((K, 1, dout)), torch.ones((K, 1, dout))).to(device = device)
            weights = torch.add(eps_w * torch.exp(0.5*self.W_v[i]), self.W_m[i])
            biases = torch.add(eps_b * torch.exp(0.5*self.b_v[i]), self.b_m[i])
            pre = torch.add(torch.einsum('mni,mio->mno', act, weights), biases)
            act = F.relu(pre)
        din = self.size[-1]
        dout = self.out_size



        eps_w = torch.normal(torch.zeros((K, din, dout)), torch.ones((K, din, dout))).to(device = device)
        eps_b = torch.normal(torch.zeros((K, 1, dout)), torch.ones((K, 1, dout))).to(device = device)
        Wtask_m = self.W_last_m[task_idx]
        Wtask_v = self.W_last_v[task_idx]
        btask_m = self.b_last_m[task_idx]
        btask_v = self.b_last_v[task_idx]

        weights = torch.add(eps_w * torch.exp(0.5*Wtask_v),Wtask_m)
        biases = torch.add(eps_b * torch.exp(0.5*btask_v), btask_m)
        act = torch.unsqueeze(act, 3)
        weights = torch.unsqueeze(weights, 1)
        pre = torch.add(torch.sum(act * weights, dim = 2), biases)
        return pre

    def _logpred(self, inputs, targets, task_idx):
        pred = self._prediction(inputs, task_idx, self.no_train_samples)
        targets = torch.unsqueeze(targets, 0).repeat([self.no_train_samples, 1, 1])
        log_liks = - (torch.nn.CrossEntropyLoss(torch.nn.softmax(pred), targets))
        log_lik = log_liks.mean()
        return log_lik


    def _KL_term(self):
        kl = 0
        for i in range(self.no_layers-1):
            din = self.size[i]
            dout = self.size[i+1]
            m, v = self.W_m[i], self.W_v[i]
            m0, v0 = self.prior_W_m[i], self.prior_W_v[i]
            const_term = -0.5 * dout * din
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            m, v = self.b_m[i], self.b_v[i]
            m0, v0 = self.prior_b_m[i], self.prior_b_v[i]
            const_term = -0.5 * dout
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

        no_tasks = len(self.W_last_m)
        din = self.size[-2]
        dout = self.size[-1]

        for i in range(no_tasks):
            m, v = self.W_last_m[i], self.W_last_v[i]
            m0, v0 = self.prior_W_last_m[i], self.prior_W_last_v[i]
            const_term = -0.5 * dout * din
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            m, v = self.b_last_m[i], self.b_last_v[i]
            m0, v0 = self.prior_b_last_m[i], self.prior_b_last_v[i]
            const_term = -0.5 * dout
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term
        return kl

    def create_head(self, head = None):
        ##TODO: check how heads of the prior are initialized
        din = self.size[-1]
        dout = self.out_size

        if head == None:
            W_m = truncated_normal([din, dout], stddev=0.1, variable = True)
            bi_m = truncated_normal([dout], stddev=0.1, variable = True)
            W_v =  init_tensor(-6.0,  dout = dout, din = din, variable = True )
            bi_v = init_tensor(-6.0, dout = dout, variable = True)

        else:

            W_m = torch.Tensor(self.W_last_m[0].data, requires_grad = True).to(device = device)
            bi_m = torch.Tensor(self.bi_last_m[0].data, requires_grad = True).to(device = device)
            W_v =  torch.Tensor(self.W_last_v[0].data, requires_grad = True).to(device = device)
            bi_v = torch.Tensor(self.bi_last_v[0].data, requires_grad = True).to(device = device)

        self.W_last_m.append(W_m)
        self.W_last_v.append(W_v)
        self.b_last_m.append(bi_m)
        self.b_last_v.append(bi_v)

        self.prior_W_last_m.append(W_m)
        self.prior_W_last_v.append(W_v)
        self.prior_b_last_m.append(bi_m)
        self.prior_b_last_v.append(bi_v)


        return

    def create_weights(self, in_dim, hidden_size, out_dim):
        hidden_size = deepcopy(hidden_size)
        hidden_size.append(out_dim)
        hidden_size.insert(0, in_dim)

        no_layers = len(hidden_size) - 1
        W_m = []
        b_m = []

        W_v = []
        b_v = []

        for i in range(no_layers-1):
            din = hidden_size[i]
            dout = hidden_size[i+1]

            #Initializiation values of means
            W_m_i= truncated_normal([din, dout], stddev=0.1, variable=True)
            bi_m_i= truncated_normal([dout], stddev=0.1, variable=True)

            #Initializiation values of variances
            W_v_i = init_tensor(-6.0,  dout = dout, din = din, variable = True)
            bi_v_i = init_tensor(-6.0,  dout = dout, variable = True)

            #Append to list weights
            W_m.append(W_m_i)
            b_m.append(bi_m_i)
            W_v.append(W_v_i)
            b_v.append(bi_v_i)

        return [W_m, b_m], [W_v, b_v]

    def create_prior(self, in_dim, hidden_size, out_dim):

        no_layers = len(hidden_size) - 1
        W_m = []
        b_m = []

        W_v = []
        b_v = []

        for i in range(no_layers - 1):
            din = hidden_size[i]
            dout = hidden_size[i + 1]

            # Initializiation values of means
            W_m_val = truncated_normal([din, dout], stddev=0.1)
            bi_m_val = truncated_normal([dout], stddev=0.1)

            # Initializiation values of variances
            W_v_val = init_tensor(np.exp(-6.0),  dout = dout, din = din )
            bi_v_val = init_tensor(np.exp(-6.0),  dout = dout, din = din )

            # Append to list weights
            W_m.append(W_m_val)
            b_m.append(bi_m_val)
            W_v.append(W_v_val)
            b_v.append(bi_v_val)

        return [W_m, b_m], [W_v, b_v]

    def update_prior(self):
        ##TODO: check if data does not detached gradient
        self.create_head(head=0)

        self.prior_W_m.data.copy_(self.W_m.data)
        self.prior_b_m.data.copy_(self.b_m.data)
        self.prior_W_last_m.data.copy_(self.W_last_m.data)
        self.prior_b_last_m.data.copy_(self.b_last_m.data)

        self.prior_W_v.data.copy_(torch.exp(self.W_v.data))
        self.prior_b_v.data.copy_(torch.exp(self.b_v.data))
        self.prior_W_last_v.data.copy_(torch.exp(self.W_last_v.data))
        self.prior_b_last_v.data.copy_(torch.exp(self.b_last_v.data))

        return