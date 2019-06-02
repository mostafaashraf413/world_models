#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 10:22:46 2019
@author: mostafa
"""

import constants
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random
from torch.utils.data import dataloader

class VAEReplayMemory(object):

    def __init__(self):
        self.capacity = constants.vae_memory_capacity
        self.memory = []

    def clear_memory(self):
        self.memory = []

    def push(self, sample):
        if len(self.memory) > self.capacity:
            self.memory = self.memory[len(self.memory) // 10 :]
        self.memory.append(sample)

    def sample(self, batch_size):
        # TODO: try another sampling methods
#        sample = random.sample(self.memory, batch_size)
#        sample = torch.FloatTensor(sample).to(constants.device)
        data = dataloader.DataLoader(dataset=self.memory, batch_size=batch_size, shuffle=True)
        return data

    def __len__(self):
        return len(self.memory)

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = torch.sigmoid(self.deconv4(x))
        return reconstruction


class Encoder(nn.Module): 
    """ VAE encoder """
    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        #self.img_size = img_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.fc_mu = nn.Linear(2*2*256, latent_size)
        self.fc_logsigma = nn.Linear(2*2*256, latent_size)


    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma
    
class VAE_Module(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, img_channels, latent_size):
        super(VAE_Module, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)
    
    def get_encoder_output(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        return z, mu, logsigma
    
    def forward(self, x): # pylint: disable=arguments-differ
        z, mu, logsigma = self.get_encoder_output(x)
        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
    
    
class VAE():
    def __init__(self, obs_dim):
        self.device = constants.device        
        self.model:VAE_Module = VAE_Module(obs_dim.shape[2], constants.vae_z_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.memory = VAEReplayMemory()
        self.__episod_loss = 0
    
    def get_z_vec(self, obs):
        self.memory.push(obs)
        obs = torch.FloatTensor(obs).unsqueeze(0).to(constants.device)
        with torch.no_grad():
            z, _, _ = self.model.get_encoder_output(obs)
        return z
    
    def optimize(self):
        if len(self.memory) < constants.vae_train_batch_size:
            return    
        batch_counter = 0
        for train_set in self.memory.sample(constants.vae_train_batch_size):
            batch_counter += 1
            train_set = train_set.type(torch.float) #torch.FloatTensor(train_set).to(constants.device)
            loss = self.__train(train_set)
            self.__episod_loss += loss
        self.__episod_loss /= batch_counter
        self.memory.clear_memory()
    
    def get_episod_loss(self):
        result = self.__episod_loss
        self.__episod_loss = 0
        return result
    
    def __train(self, dataset_train):
        """ One training epoch """
        self.model.train()
#        for batch_idx, data in enumerate(train_loader):
        data = dataset_train.to(self.device)
        self.optimizer.zero_grad()
        recon_batch, mu, logvar = self.model(data)
        loss = self.loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logsigma):
        """ VAE loss function """
        BCE = F.mse_loss(recon_x, x, reduction='sum')#size_average=False)
    
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
        return BCE + KLD

    
    def save(self):
        pass
    
    def load():
        pass