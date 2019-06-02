#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 10:22:58 2019

@author: mostafa
"""
import constants
import torch
import random

import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal
import numpy as np

class MDRNNReplayMemory(object):

    def __init__(self):
        self.capacity = constants.mdn_rnn_memory_capacity
        self.memory = []

    def clear_memory(self):
        self.memory = []

    def push(self, sample):
        if len(self.memory) > self.capacity:
            self.memory = self.memory[len(self.memory) // 10 :]
        self.memory.append(sample)
        
    def sample(self, batch_size):        
        latent_obs , action, reward, terminal, latent_next_obs = [], [], [], [], []
        
        for i in range(len(self.memory) - constants.mdn_rnn_seq_length - 1):
            obs_seq = []
            action_seq = []
            reward_seq = []
            terminal_seq = []
            for j in range(i, i + constants.mdn_rnn_seq_length - 1):
                obs_seq.append(self.memory[j][ : constants.vae_z_size])
                action_seq.append(self.memory[j][constants.vae_z_size : -2])
                reward_seq.append(self.memory[j][-2])
                terminal_seq.append(self.memory[j][-1])
                
            latent_obs.append(obs_seq)
            action.append(action_seq)
            reward.append(reward_seq)
            terminal.append(terminal_seq)            
            latent_next_obs.append(self.memory[i + constants.mdn_rnn_seq_length][ : constants.vae_z_size])
            
            
        latent_obs, action, reward, terminal, latent_next_obs = [torch.tensor(arr) for arr in 
                                                                   [latent_obs, action, reward, terminal, latent_next_obs]]
        terminal = terminal.type(torch.FloatTensor).to(constants.device)
        
        latent_next_obs.unsqueeze(1) 
        
        rand_perm = torch.randperm(latent_obs.shape[0])
        
        from_, to = 0, batch_size
        while from_ < rand_perm.shape[0]:
            batch_idxs = rand_perm[from_ : to]
            from_ = to
            to += batch_size
        
            latent_obs_batch, action_batch, reward_batch, terminal_batch, latent_next_obs_batch = [t[batch_idxs] for t in 
                                                                   [latent_obs, action, reward, terminal, latent_next_obs]]       
        
            latent_obs_batch, action_batch, reward_batch, terminal_batch = [arr.transpose(1,0) for arr in [latent_obs_batch, action_batch, reward_batch, terminal_batch]]
            
            latent_obs_batch, action_batch, reward_batch, terminal_batch, latent_next_obs_batch = [t.to(constants.device) for t in 
                                                                                                   [latent_obs_batch, action_batch, reward_batch, terminal_batch, latent_next_obs_batch]]
            
            yield latent_obs_batch, action_batch, reward_batch, terminal_batch, latent_next_obs_batch


    def __len__(self):
        return len(self.memory)


class _MDRNNBase(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = nn.Linear(
            hiddens, (2 * latents + 1) * gaussians + 2)

    def forward(self, *inputs):
        pass

class MDRNN(_MDRNNBase):
    """ MDRNN model for multi steps forward """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTM(latents + actions, hiddens)

    def forward(self, actions, latents): # pylint: disable=arguments-differ
        """ MULTI STEPS forward.
        :args actions: (SEQ_LEN, BSIZE, ASIZE) torch tensor
        :args latents: (SEQ_LEN, BSIZE, LSIZE) torch tensor
        :returns: mu_nlat, sig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
            - rs: (SEQ_LEN, BSIZE) torch tensor
            - ds: (SEQ_LEN, BSIZE) torch tensor
        """
        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins)
        gmm_outs = self.gmm_linear(outs)

        stride = self.gaussians * self.latents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussians, self.latents)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        
#        logpi = f.log_softmax(pi, dim=-1)
        logpi = f.softmax(pi, dim=-1)
        
        rs = gmm_outs[:, :, -2]

        ds = gmm_outs[:, :, -1]

        return mus, sigmas, logpi, rs, ds, outs[-1]


class MDN_RNN():
    def __init__(self, action_size):
        self.mdrnn = MDRNN(constants.vae_z_size, action_size, constants.mdn_rnn_hidden_size, constants.n_gaussians)
        self.mdrnn.to(constants.device)
        self.memory = MDRNNReplayMemory()
        self.optimizer = torch.optim.RMSprop(self.mdrnn.parameters(), lr=1e-3, alpha=.9)
        self.action_size = action_size
        
        self.__episod_total_loss = 0
        self.__episod_gmm_loss = 0
        self.__episod_reward_loss = 0
        self.__episod_terminal_loss = 0
        
    def __action_to_vec(self, action:int):
        action_vec = torch.zeros(self.action_size)
        action_vec[action] = 1
        return action_vec
    
    def get_episod_loss(self):
        result = '{total_loss: %f, gmm_loss: %f, reward_loss: %f, terminal_loss: %f}' %(self.__episod_total_loss,
                      self.__episod_gmm_loss, self.__episod_reward_loss, self.__episod_terminal_loss)
        
        self.__episod_total_loss = 0
        self.__episod_gmm_loss = 0
        self.__episod_reward_loss = 0
        self.__episod_terminal_loss = 0
        
        return result
    
    def get_hidden_vec(self, action, z_vec):
        with torch.no_grad():
            action_vec = self.__action_to_vec(action)
            action_vec = action_vec.unsqueeze(0).unsqueeze(0).to(constants.device)
            
            z_vec = z_vec.unsqueeze(0).to(constants.device)
            _, _, _, _, _, hidden = self.mdrnn(action_vec, z_vec)
        
        return hidden
    
    def insert_training_sample(self, obs:torch.Tensor, action:int, reward:int, done):
        action_vec = self.__action_to_vec(action)
        obs = obs.squeeze(0)
        sample = obs.tolist() + action_vec.tolist()
        sample.append(reward)
        sample.append(done)
        self.memory.push(sample)
    
    def optimize(self):
        if len(self.memory.memory) < constants.mdn_rnn_train_batch_size:
            return     
        batch_counter = 0
        for latent_obs, action, reward, terminal, latent_next_obs in self.memory.sample(constants.mdn_rnn_train_batch_size):
            batch_counter+=1       
            losses = self.__get_loss(latent_obs, action, reward, terminal, latent_next_obs, include_reward = constants.mdn_rnn_include_reward)
    
            self.optimizer.zero_grad()
            losses['loss'].backward()
            self.optimizer.step()
            
            self.__episod_total_loss += losses['loss'].item()
            self.__episod_gmm_loss += losses['gmm'].item()
            self.__episod_reward_loss += losses['mse'].item()
            self.__episod_terminal_loss += losses['bce'].item()
            
        self.__episod_total_loss /= batch_counter
        self.__episod_gmm_loss /= batch_counter
        self.__episod_reward_loss /= batch_counter
        self.__episod_terminal_loss /= batch_counter
        self.memory.clear_memory()
    
    def initial_state(self):
        return torch.rand(constants.mdn_rnn_hidden_size).unsqueeze(0).to(constants.device)
    
    def __gmm_loss(self, batch, mus, sigmas, logpi, reduce=True): # pylint: disable=too-many-arguments
        """ Computes the gmm loss.
        Compute minus the log probability of batch under the GMM model described
        by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
        dimensions (several batch dimension are useful when you have both a batch
        axis and a time step axis), gs the number of mixtures and fs the number of
        features.
        :args batch: (bs1, bs2, *, fs) torch tensor
        :args mus: (bs1, bs2, *, gs, fs) torch tensor
        :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
        :args logpi: (bs1, bs2, *, gs) torch tensor
        :args reduce: if not reduce, the mean in the following formula is ommited
        :returns:
        loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
            sum_{k=1..gs} pi[i1, i2, ..., k] * N(
                batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))
        NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
        with fs).
        """
        batch = batch.unsqueeze(-2)
        normal_dist = Normal(mus[-1], sigmas[-1])
        g_log_probs = normal_dist.log_prob(batch)
        g_log_probs = logpi[-1] + torch.sum(g_log_probs, dim=-1)
#        max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
#        g_log_probs = g_log_probs - max_log_probs
    
#        g_probs = torch.exp(g_log_probs)
#        probs = torch.sum(g_probs, dim=-1)
    
#        log_prob = max_log_probs.squeeze() + torch.log(probs)
        
#        if reduce:
#            return - torch.mean(log_prob)
#        return - log_prob
        
        if reduce:
            return - torch.mean(g_log_probs)
        return - g_log_probs
        

    def __get_loss(self, latent_obs, action, reward, terminal,
                 latent_next_obs, include_reward: bool = True):
        """ Compute losses.
        The loss that is computed is:
        (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
             BCE(terminal, logit_terminal)) / (LSIZE + 2)
        The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
        approximately linearily with LSIZE. All losses are averaged both on the
        batch and the sequence dimensions (the two first dimensions).
        :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
        :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
        :args reward: (BSIZE, SEQ_LEN) torch tensor
        :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
        :returns: dictionary of losses, containing the gmm, the mse, the bce and
            the averaged loss.
        """
        
        mus, sigmas, logpi, rs, ds, _ = self.mdrnn(action, latent_obs)
        gmm = self.__gmm_loss(latent_next_obs, mus, sigmas, logpi)
        bce = f.binary_cross_entropy_with_logits(ds, terminal)
        if include_reward:
            mse = f.mse_loss(rs, reward)
            scale = constants.vae_z_size + 2
        else:
            mse = 0
            scale = constants.vae_z_size + 1
        loss = (gmm + bce + mse) / scale
        return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)
    
    def save(self):
        pass
    
    def load():
        pass