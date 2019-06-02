#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 23:33:18 2019

@author: mostafa

- Original model sourse: "https://github.com/ctallec/world-models" 
"""

import argparse
import cv2

from controller import Controller
from VAE import VAE
from MDN_RNN import MDN_RNN
import constants
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gym
import torchvision.transforms as T
from PIL import Image
import torch


#def preprocess_obs(screen):
#    # Returned screen requested by gym is 400x600x3, but is sometimes larger
#    # such as 800x1200x3. Transpose it into torch order (CHW).
#    screen = cv2.resize(screen, (64, 64))
#    screen = screen.transpose((2, 0, 1))
#    #normalize
#    screen = screen / 255.0
#    return  screen

resize = T.Compose([T.ToPILImage(),
                    T.Resize(64, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    
    screen = screen.transpose((1, 2, 0))
    screen = cv2.resize(screen, (64, 64))
    screen = screen.transpose((2, 1, 0))

#    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)

    return screen #.unsqueeze(0).to(device)



def run_episode(env, vae:VAE, mdn_rnn:MDN_RNN, ctrl:Controller):
    done = False
    comulative_reward = 0.0
    env.reset()
    
#    obs = env.render(mode = 'rgb_array')
#    obs = preprocess_obs(obs)
    obs = get_screen()
    
    ctrl_idx = ctrl.get_current_controller_idx() 
    hidden_vec = mdn_rnn.initial_state()
    
    while not done:
        obs_z_vec = vae.get_z_vec(obs)
        action = ctrl.get_action(obs_z_vec, hidden_vec)
        hidden_vec = mdn_rnn.get_hidden_vec(action, obs_z_vec)
        
        _, reward, done, _ = env.step(action)
#        _, reward, done, _ = env.step(env.action_space.sample())
        
#        obs = env.render(mode = 'rgb_array')       
#        obs = preprocess_obs(obs)

        obs = get_screen()
        
        comulative_reward += reward
        mdn_rnn.insert_training_sample(obs_z_vec, action, reward, done)
        
    ctrl.insert_evluation(comulative_reward)
#    print('episode has been finished, total rewards: %f, controller: %d, generation: %d' %(comulative_reward, ctrl_idx, ctrl.get_generation_number()))
    
    if ctrl_idx == (constants.controller_pop_size - 1):
        print('start training VAE')
        vae.optimize()
        print('start training MDN_RNN')
        mdn_rnn.optimize()
        print('start training controller CMA')
        ctrl.optimize()
        print('Generatin: %d ---> VAE loss: %f, MDN_RNN loss: %s, generation_average_rewards: %f' %(ctrl.get_generation_number(), vae.get_episod_loss(), mdn_rnn.get_episod_loss(), 
                                                                                 (ctrl.get_current_total_eval() / constants.controller_pop_size)))
        print('################################################################################')
    
    return comulative_reward



def run_evaluation(env):
    while not env.done_grading():
        run_episode(env)
        env.reset()

if __name__ == '__main__':
        
    try:
        env = gym.make('CartPole-v0').unwrapped

        # set up matplotlib
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display

        plt.ion()
    
        env.reset()
        # initialize world models
        vae = VAE(env.render(mode = 'rgb_array'))
        mdn_rnn = MDN_RNN(env.action_space.n)
        ctrl = Controller(env.action_space.n)

        while True:
            episode_reward = run_episode(env, vae, mdn_rnn, ctrl)
#                print('Episode total reward: %f' %(episode_reward))
                
    finally:
        env.close()