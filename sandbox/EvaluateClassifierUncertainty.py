import torch
from torch import nn
import numpy as np
import torch.nn.modules
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATH = './classifier_params'

class Classifier(nn.Module):
    dimX = 28*28
    dimH = 1000
    num_classes = 10
    dropout_p = 0.2
    batch_size = 100
    epochs = 500

    @classmethod
    def lin_with_dropout(cls, d_in, d_out):
        return nn.Sequential( \
            nn.Linear(d_in, d_out).to(device=device), \
            nn.modules.ReLU(), \
            nn.Dropout(p=Classifier.dropout_p))

    def __init__(self):
        super(Classifier, self).__init__()
        self.net = nn.Sequential( \
            Classifier.lin_with_dropout(Classifier.dimX, Classifier.dimH), \
            Classifier.lin_with_dropout(Classifier.dimH, Classifier.dimH), \
            Classifier.lin_with_dropout(Classifier.dimH, Classifier.dimH), \
            nn.Linear(Classifier.dimH, Classifier.num_classes).to(device=device),
            nn.Softmax(dim=1))


    def train_model(self, dataset):
        """
        :param dataset: torchvision.dataset object (MNIST or NotMNIST)
        :param n_epochs:
        :return: nothing
        """
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=Classifier.batch_size, shuffle=True, num_workers=2)

        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters())

        for _ in range(Classifier.epochs):
            for i, batch in enumerate(trainloader):
                data, labels = batch
                data = data.to(device=device)
                labels = labels.to(device=device)
                optimizer.zero_grad()
                batch_loss = loss(self(data.view(-1,Classifier.dimX)), labels)
                batch_loss.backward()
                optimizer.step()
                print('Batch ', i, batch_loss.item())



    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load_model(cls, path):
        c = Classifier()
        c.load_state_dict(torch.load(path,map_location=device))
        c.eval()
        return c

    def forward(self, x):
        return self.net(x)


class EvaluateClassifierUncertainty:
    def __init__(self, classifier_param_load_path, should_print = True):
        self.classifier = Classifier.load_model(classifier_param_load_path).to(device=device)
        self.should_print = should_print

    def __call__(self, task_id, task_model, loader):
        dimZ = 50
        samples_per_iter = 100
        num_iter = 10
        loss_mu = 0.0
        loss_var = 0.0
        for _ in range(num_iter):
            Zs_params = torch.zeros(samples_per_iter, dimZ*2, device=device)
            reconstructed_Xs = task_model.sample_and_decode(Zs_params)
            true_Ys = torch.ones(samples_per_iter, dtype=torch.long, device=device) * task_id # these are the labels for the generated pictures
            cross_entropies = F.cross_entropy(self.classifier(reconstructed_Xs), true_Ys, reduction='none')
            loss_mu += torch.mean(cross_entropies) / num_iter
            loss_var += torch.mean((cross_entropies - loss_mu)**2) / num_iter

        if self.should_print:
            print("test_classifier=%.2f, std=%.2f" \
                  % (loss_mu, np.sqrt(loss_var.cpu() / (num_iter*samples_per_iter))))
        return loss_mu, np.sqrt(loss_var.cpu() / (num_iter*samples_per_iter))


if __name__ == '__main__':
    c = Classifier().to(device=device)
    #c = Classifier.load_model(PATH)
    c.train_model(torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor()))
    c.save_model(PATH)


