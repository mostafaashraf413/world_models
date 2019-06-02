#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 10:23:35 2019

@author: mostafa
"""

import constants
import cma

import torch
import torch.nn as nn
import numpy as np

from utils import flatten_parameters, unflatten_parameters

"""
Tips:
 - population size of 64.
 - The agent’s fitness value is the average cumulative reward of the 16 random rollouts.   
"""

class Controller_Module(nn.Module):
    """ Controller """
    def __init__(self, latents, recurrents, actions):
        super().__init__()
        self.fc = nn.Linear(latents + recurrents, actions)

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=-1)
        x = self.fc(cat_in)
        x = torch.softmax(x, dim=1)
        return x

class Controller:
    def __init__(self, action_space_size):
        
        latents_size = constants.vae_z_size
        recurrents_size = constants.mdn_rnn_hidden_size
        
        self.device = constants.device
        self.controller_model = Controller_Module(latents_size, recurrents_size, action_space_size).to(self.device)
        
        for param in self.controller_model.parameters():
            param.requires_grad = False
            
        init_sol = flatten_parameters(self.controller_model.parameters())
        self.__cma_es = cma.CMAEvolutionStrategy(init_sol, 1.0, {'popsize': constants.controller_pop_size})
#        self.__cma_es = cma.purecma.CMAES(init_sol, 0.9, popsize = constants.controller_pop_size)
        
        self.__reinitialize_params()
        self.__generation_number = 0
        self.__current_total_eval = 0
        
    def get_current_total_eval(self):
        return self.__current_total_eval
    
    def __reinitialize_params(self):
        self.__controllers_weights_lst = self.__cma_es.ask()
        self.__controllers_evaluations_lst = [0]*constants.controller_pop_size
        self.__current_controller_idx = 0
    
    def get_action(self, z_vec, hidden_vec):
        
        with torch.no_grad():
            actions = self.controller_model(z_vec, hidden_vec)[0]
            _, best_action = actions.max(0)
            best_action = best_action.item()
#            print(best_action)
        return best_action
    
    def insert_evluation(self, com_rewards):
        # change the reward sign to convert the problem to minimization
        self.__controllers_evaluations_lst[self.__current_controller_idx] = -com_rewards
        self.__current_controller_idx += 1
            
        # change current model weights
        if self.__current_controller_idx < constants.controller_pop_size:
            weights = unflatten_parameters(self.__controllers_weights_lst[self.__current_controller_idx], 
                                           self.controller_model.parameters(), constants.device)
            self.controller_model.fc.weight[:] = weights[0]
            self.controller_model.fc.bias[:] = weights[1]
            
    
    def get_current_controller_idx(self):
        if self.__current_controller_idx >= constants.controller_pop_size:
            raise IndexError('controller index is larger than pop size!!')
        return self.__current_controller_idx
    
    def get_generation_number(self):
        return self.__generation_number
    
    def optimize(self):
        self.__cma_es.tell(self.__controllers_weights_lst, self.__controllers_evaluations_lst)
        self.__generation_number += 1
        self.__current_total_eval = - np.sum(self.__controllers_evaluations_lst)
        # reinitialize
        self.__reinitialize_params()
    
    def save(self):
        pass
    
    def load():
        pass