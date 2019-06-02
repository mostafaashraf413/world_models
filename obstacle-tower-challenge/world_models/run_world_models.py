#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 23:33:18 2019

@author: mostafa

- Original model sourse: "https://github.com/ctallec/world-models" 
"""

from obstacle_tower_env import ObstacleTowerEnv
import argparse
import cv2

from controller import Controller
from VAE import VAE
from MDN_RNN import MDN_RNN
import constants


def preprocess_obs(screen):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = cv2.resize(screen, (64, 64))
    screen = screen.transpose((2, 0, 1))
    #normalize
    screen = screen / 255.0
    return  screen

def run_episode(env, vae:VAE, mdn_rnn:MDN_RNN, ctrl:Controller):
    done = False
    comulative_reward = 0.0
    obs = env.reset()
    obs = preprocess_obs(obs)
    ctrl_idx = ctrl.get_current_controller_idx() 
    hidden_vec = mdn_rnn.initial_state()
    
    while not done:
        obs_z_vec = vae.get_z_vec(obs)
        action = ctrl.get_action(obs_z_vec, hidden_vec)
        hidden_vec = mdn_rnn.get_hidden_vec(action, obs_z_vec)
        
        obs, reward, done, info = env.step(action)
        obs = preprocess_obs(obs)
        comulative_reward += reward
        mdn_rnn.insert_training_sample(obs_z_vec, action, reward, done)
        
    ctrl.insert_evluation(comulative_reward)
    print('episode has been finished, total rewards: %f, controller: %d, generation: %d' %(comulative_reward, ctrl_idx, ctrl.get_generation_number()))
    
    if ctrl_idx == (constants.controller_pop_size - 1):
        print('start training VAE')
        vae.optimize()
        print('start training MDN_RNN')
        mdn_rnn.optimize()
        print('start training controller CMA')
        ctrl.optimize()
        print('VAE loss: %f, MDN_RNN loss: %s, generation_total_rewards: %f' %(vae.get_episod_loss(), mdn_rnn.get_episod_loss(), ctrl.get_current_total_eval()))
        print('################################################################################')
    
    return comulative_reward

def run_evaluation(env):
    while not env.done_grading():
        run_episode(env)
        env.reset()

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('environment_filename', default='./../ObstacleTower/obstacletower', nargs='?')
    parser.add_argument('--docker_training', action='store_true')
    parser.set_defaults(docker_training=False)
    args = parser.parse_args()
    
    try:
        env = ObstacleTowerEnv(args.environment_filename, docker_training=args.docker_training, realtime_mode=False)
    
        # initialize world models
        vae = VAE(env.observation_space)
        mdn_rnn = MDN_RNN(env.action_space.n)
        ctrl = Controller(env.action_space.n)
    
    
        if env.is_grading():
            episode_reward = run_evaluation(env)
        else:
            while True:
                episode_reward = run_episode(env, vae, mdn_rnn, ctrl)
#                print('Episode total reward: %f' %(episode_reward))
                
    finally:
        env.close()