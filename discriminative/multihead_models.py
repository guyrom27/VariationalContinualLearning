import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import truncnorm
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(0)
#tf.set_random_seed(0)

# variable initialization functions
def truncated_normal(size, std=1):
    values = truncnorm.rvs(-2 * std, 2 * std, size=size)
    x = torch.from_numpy(values).device(device)
    return x

class Cla_NN(object):
    def train(self, x_train, y_train, task_idx, no_epochs=1000, batch_size=100, display_epoch=5):
        N = x_train.shape[0]
        if batch_size > N:
            batch_size = N

        costs = []
        # Training cycle
        for epoch in range(no_epochs):
            perm_inds = range(x_train.shape[0])
            np.random.shuffle(perm_inds)
            cur_x_train = x_train[perm_inds]
            cur_y_train = y_train[perm_inds]

            avg_cost = 0.
            total_batch = int(np.ceil(N * 1.0 / batch_size))
            # Loop over all batches
            for i in range(total_batch):
                start_ind = i*batch_size
                end_ind = np.min([(i+1)*batch_size, N])
                batch_x = cur_x_train[start_ind:end_ind, :]
                batch_y = cur_y_train[start_ind:end_ind, :]
                self._prediction()
                # # Run optimization op (backprop) and cost op (to get loss value)
                # _, c = sess.run(
                #     [self.train_step, self.cost],
                #     feed_dict={self.x: batch_x, self.y: batch_y, self.task_idx: task_idx})

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
    def __init__(self, input_size, hidden_size, output_size, training_size, prev_weights=None, learning_rate=0.001):
        #
        super(Vanilla_NN, self).__init__(input_size, hidden_size, output_size, training_size)
        # # init weights and biases
        self.W, self.b, self.W_last, self.b_last, self.size = self.create_weights(
                 input_size, hidden_size, output_size, prev_weights)
        self.no_layers = len(hidden_size) + 1
        #self.pred = self._prediction(self.x, self.task_idx)
        #self.cost = - self._logpred(self.x, self.y, self.task_idx)
        self.weights = [self.W, self.b, self.W_last, self.b_last]

    def _prediction(self, inputs, task_idx):
        act = inputs
        for i in range(self.no_layers-1):
             pre = torch.add(torch.matmul(act, self.W[i]), self.b[i])
             act = torch.nn.functional.relu(pre)
        pre = torch.add(torch.matmul(act, self.W_last[task_idx]), self.b_last[task_idx])
        return pre

    def _logpred(self, inputs, targets, task_idx):
        pred = self._prediction(inputs, task_idx)
        log_liks = - (torch.nn.CrossEntropyLoss(torch.nn.softmax(pred), targets))
        log_lik = log_liks.mean()
        return log_lik

    def get_loss(self, batch_x, batch_y, task_idx):
        return self._logpred(self, batch_x, batch_y, task_idx)

    def create_weights(self, in_dim, hidden_size, out_dim, prev_weights):
        # hidden_size = deepcopy(hidden_size)
        # hidden_size.append(out_dim)
        # hidden_size.insert(0, in_dim)
        # no_params = 0
        # no_layers = len(hidden_size) - 1
        # W = []
        # b = []
        # W_last = []
        # b_last = []
        # for i in range(no_layers-1):
        #     din = hidden_size[i]
        #     dout = hidden_size[i+1]
        #     if prev_weights is None:
        #         Wi_val = tf.truncated_normal([din, dout], stddev=0.1)
        #         bi_val = tf.truncated_normal([dout], stddev=0.1)
        #     else:
        #         Wi_val = tf.constant(prev_weights[0][i])
        #         bi_val = tf.constant(prev_weights[1][i])
        #     Wi = tf.Variable(Wi_val)
        #     bi = tf.Variable(bi_val)
        #     W.append(Wi)
        #     b.append(bi)
        #
        # if prev_weights is not None:
        #     prev_Wlast = prev_weights[2]
        #     prev_blast = prev_weights[3]
        #     no_prev_tasks = len(prev_Wlast)
        #     for j in range(no_prev_tasks):
        #         W_j = prev_Wlast[j]
        #         b_j = prev_blast[j]
        #         Wi = tf.Variable(W_j)
        #         bi = tf.Variable(b_j)
        #         W_last.append(Wi)
        #         b_last.append(bi)
        #
        # din = hidden_size[-2]
        # dout = hidden_size[-1]
        # Wi_val = tf.truncated_normal([din, dout], stddev=0.1)
        # bi_val = tf.truncated_normal([dout], stddev=0.1)
        # Wi = tf.Variable(Wi_val)
        # bi = tf.Variable(bi_val)
        # W_last.append(Wi)
        # b_last.append(bi)
        #
        # return W, b, W_last, b_last, hidden_size
        return

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
        self.size = hidden_size

        self.W_m, self.b_m = m1[0], m1[1]
        self.W_v, self.b_v = v1[0], v1[1]

        self.W_last_m, self.b_last_m = [], []

        m2, v2 = self.create_prior(input_size, hidden_size, output_size, prev_means, prev_log_variances, prior_mean, prior_var)

        self.prior_W_m, self.prior_b_m, = m2[0], m2[1]
        self.prior_W_v, self.prior_b_v = v2[0], v2[1]

        self.prior_W_last_m, self.prior_b_last_m = [], []
        self.prior_W_last_v, self.prior_b_last_v = [], []


        self.create_head()

        ##append the last layers to the general weights to keep track of the gradient easily
        m1.append(self.W_last_m)
        m1.append(self.bi_last_m)
        v1.append(self.W_last_v)
        v1.append(self.bi_last_v)

        self.weights = [m1, v1]

        self.no_layers = len(self.size) - 1
        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples
        #self.pred = self._prediction(self.x, self.task_idx, self.no_pred_samples)
        #self.cost = torch.div(self._KL_term(), training_size) - self._logpred(self.x, self.y, self.task_idx)
        self.training_size = training_size
        self.optimizer = optim.SGD(self.weights, lr=0.01)

    def get_loss(self, batch_x, batch_y, task_idx):
        return torch.div(self._KL_term(), self.training_size) - self._logpred(batch_x, batch_y, task_idx)

    def _prediction(self, inputs, task_idx, no_samples):
        return self._prediction_layer(inputs, task_idx, no_samples)

    # this samples a layer at a time
    def _prediction_layer(self, inputs, task_idx, no_samples):
        K = no_samples
        act = torch.unsqueeze(inputs, 0).repeat([K, 1, 1])
        for i in range(self.no_layers-1):
            din = self.size[i]
            dout = self.size[i+1]
            eps_w = torch.normal(torch.zeros((K, din, dout)), torch.ones((K, din, dout)))
            eps_b = torch.normal(torch.zeros((K, 1, dout)), torch.ones((K, 1, dout)))
            weights = torch.add(torch.mm(eps_w, torch.exp(0.5*self.W_v[i])), self.W_m[i])
            biases = torch.add(torch.mm(eps_b, torch.exp(0.5*self.b_v[i])), self.b_m[i])
            pre = torch.add(torch.einsum('mni,mio->mno', act, weights), biases)
            act = F.relu(pre)
        din = self.size[-2]
        dout = self.size[-1]
        eps_w = torch.normal(torch.zeros((K, din, dout)), torch.ones((K, din, dout)))
        eps_b = torch.normal(torch.zeros((K, 1, dout)), torch.ones((K, 1, dout)))

        Wtask_m = self.W_last_m[task_idx]
        Wtask_v = self.W_last_v[task_idx]
        btask_m = self.b_last_m[task_idx]
        btask_v = self.b_last_v[task_idx]
        weights = torch.add(torch.mm(eps_w, torch.exp(0.5*Wtask_v)), Wtask_m)
        biases = torch.add(torch.mm(eps_b, torch.exp(0.5*btask_v)), btask_m)
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
            log_std_diff = 0.5 * torch.sum(np.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            m, v = self.b_m[i], self.b_v[i]
            m0, v0 = self.prior_b_m[i], self.prior_b_v[i]
            const_term = -0.5 * dout
            log_std_diff = 0.5 * torch.sum(np.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

        no_tasks = len(self.W_last_m)
        din = self.size[-2]
        dout = self.size[-1]

        for i in range(no_tasks):
            m, v = self.W_last_m[i], self.W_last_v[i]
            m0, v0 = self.prior_W_last_m[i], self.prior_W_last_v[i]
            const_term = -0.5 * dout * din
            log_std_diff = 0.5 * torch.sum(np.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            m, v = self.b_last_m[i], self.b_last_v[i]
            m0, v0 = self.prior_b_last_m[i], self.prior_b_last_v[i]
            const_term = -0.5 * dout
            log_std_diff = 0.5 * torch.sum(np.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term
        return kl

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
            W_m_val = truncated_normal([din, dout], stddev=0.1)
            bi_m_val = truncated_normal([dout], stddev=0.1)

            #Initializiation values of variances
            W_v_val = torch.Tensor(-6.0, shape=[din, dout], requires_grad = False).device(device)
            bi_v_val = torch.Tensor(-6.0, shape=[dout], requires_grad = False).device(device)

            W_m = Variable(W_m_val)
            bi_m = Variable(bi_m_val)
            W_v = Variable(W_v_val)
            bi_v = Variable(bi_v_val)

            #Append to list weights
            W_m.append(W_m)
            b_m.append(bi_m)
            W_v.append(W_v)
            b_v.append(bi_v)
        return [W_m, b_m], [W_v, b_v]

    def create_head(self, head = None):
        ##TODO: check how heads of the prior are initialized
        din = self.size[-1]
        dout = self.out_dim

        if head == None:
            W_m_val = truncated_normal([din, dout], stddev=0.1)
            bi_m_val = truncated_normal([dout], stddev=0.1)
            W_v_val = torch.Tensor(-6.0, shape=[din, dout], requires_grad=False).device(device)
            bi_v_val = torch.Tensor(-6.0, shape=[dout], requires_grad=False).device(device)

        else:

            W_m_val = self.W_last_m[0].data
            bi_m_val = self.bi_last_m[0].data
            W_v_val =  self.W_last_m[0].data
            bi_v_val = self.bi_last_v[0].data

        self.W_last_m.append(Variable(W_m_val))
        self.W_last_v.append(Variable(W_v_val))
        self.bi_last_m.append(Variable(bi_m_val))
        self.bi_last_v.append(Variable(bi_v_val))

        self.prior_W_last_m.append(W_m_val)
        self.prior_W_last_v.append(W_v_val)
        self.prior_bi_last_m.append(bi_m_val)
        self.prior_bi_last_v.append(bi_v_val)


        return

    def update_prior(self):

        self.create_head(head=0)

        self.prior_W_m.data.copy_(self.W_m.data)
        self.prior_b_m.data.copy_(self.b_m.data)
        self.prior_W_last_m.data.copy_(self.W_last_m.data)
        self.prior_b_last_m.data.copy_(self.b_last_m.data)

        self.prior_W_v.data.copy_(self.W_v.data)
        self.prior_b_v.data.copy_(self.b_v.data)
        self.prior_W_last_v.data.copy_(self.W_last_v.data)
        self.prior_b_last_v.data.copy_(self.b_last_v.data)

        return