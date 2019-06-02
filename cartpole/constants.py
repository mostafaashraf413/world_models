#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 10:45:52 2019

@author: mostafa
"""
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VAE parameters
vae_model_file_name = 'vae.model'
vae_z_size = 64
vae_memory_capacity = 100000
vae_train_batch_size = 128

# MDN_RNN parameters
mdn_rnn_model_file_name = 'mdn_rnn.model'
n_gaussians = 5
mdn_rnn_hidden_size = 512
mdn_rnn_memory_capacity = 100000
mdn_rnn_train_batch_size = 128
mdn_rnn_seq_length = 50
mdn_rnn_include_reward = True

# Controller parameters
controller_model_file_name = 'controller.model'
controller_pop_size = 64




