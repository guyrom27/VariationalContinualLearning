import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
opt, unknown = parser.parse_known_args()
opt.n_epochs = 10
opt.batch_size = 256
opt.lr = 0.0002
opt.b1 = 0.5
opt.b2 = 0.999
opt.n_cpu = 8
opt.latent_dim = 100
opt.num_classes = 10
opt.img_size = 28
opt.channels = 1
opt.sample_interval = 400
print(opt)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor



class VGR():
    def __init__(self, task_id):
        self.task_id = task_id
        # Loss functions
        self.adversarial_loss = torch.nn.BCELoss()
        self.auxiliary_loss = torch.nn.CrossEntropyLoss()
        # Initialize generator and discriminator
        self.generator = Generator()
        self.discriminator = Discriminator()
        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()
            self.auxiliary_loss.cuda()

        # Initialize weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        # Configure data loader
        #os.makedirs('../../data/mnist', exist_ok=True)
        #self.dataloader = torch.utils.data.DataLoader(
        #    datasets.MNIST('../../data/mnist', train=True, download=True,
        #                   transform=transforms.Compose([
        #                        transforms.ToTensor(),
        #                        transforms.Normalize([0.5], [0.5])
        #                   ])),
        #    batch_size=opt.batch_size, shuffle=True)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


    def train(self, x_train, y_train):
        N = x_train.shape[0]
        for epoch in range(opt.n_epochs):

            total_batch = int(np.ceil(N * 1.0 / opt.batch_size))
            # Loop over all batches
            for i in range(total_batch):
                start_ind = i*opt.batch_size
                end_ind = np.min([(i+1)*opt.batch_size, N])
                batch_x = torch.Tensor(x_train[start_ind:end_ind, :]).to(device = device)
                batch_y = torch.Tensor(y_train[start_ind:end_ind]).to(device = device)
                batch_x = batch_x.reshape(-1,opt.img_size,opt.img_size).unsqueeze(1)
                batch_size = batch_x.shape[0]

                # Adversarial ground truths
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(batch_x.type(FloatTensor))
                labels = Variable(batch_y.type(LongTensor))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                validity, _ = self.discriminator(gen_imgs)
                g_loss = self.adversarial_loss(validity, valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Loss for real images
                real_pred, real_aux = self.discriminator(real_imgs)
                d_real_loss =  (self.adversarial_loss(real_pred, valid) + self.auxiliary_loss(real_aux, labels)) / 2

                # Loss for fake images
                fake_pred, fake_aux = self.discriminator(gen_imgs.detach())
                d_fake_loss = self.adversarial_loss(fake_pred, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                # Calculate discriminator accuracy
                pred = real_aux.data.cpu().numpy()
                gt = labels.data.cpu().numpy()
                d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                d_loss.backward()
                self.optimizer_D.step()

                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]" % (epoch, opt.n_epochs, i, total_batch,
                                                                    d_loss.item(), 100 * d_acc,
                                                                    g_loss.item()))

                #batches_done = epoch * len(self.dataloader) + i
                #if batches_done % opt.sample_interval == 0:
                #    save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)



    def generate_samples(self, no_samples):
         # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (no_samples, opt.latent_dim))))
        # Generate a batch of images
        gen_imgs = self.generator(z)
        _,labels = self.discriminator(gen_imgs)
        return gen_imgs,labels

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = opt.img_size // 4 # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # Output layers
        self.adv_layer = nn.Sequential( nn.Linear(512, 1),
                                        nn.Sigmoid())
        self.aux_layer = nn.Sequential( nn.Linear(512, opt.num_classes+1),
                                        nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


