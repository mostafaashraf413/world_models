#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 06:01:49 2019

@author: mostafa
"""

from obstacle_tower_env import ObstacleTowerEnv
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import cv2
from collections import namedtuple
from itertools import count
import random
import numpy as np
import math
import matplotlib.pyplot as plt

def displayImage(image):
    plt.imshow(image)
    plt.show()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = torch.FloatTensor(screen)
    # Resize, and add a batch dimension (BCHW)
    screen = screen.unsqueeze(0).to(device)
    #normalize
    screen = screen / 255.0
    return  screen

class DQN(nn.Module):

    def __init__(self, h, w, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, n_actions) # 448 or 512

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


steps_done = 0
def select_action(state, n_actions):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
#    print('loss = ',loss.item())
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def run_episode(env, i_episode, TARGET_UPDATE):
#    done = False
#    reward = 0.0
#    
#    while not done:
##        action = env.action_space.sample()
#        obs, reward, done, info = env.step(action)
#    return reward
    
    total_episode_reward = 0
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen(env)#env.render(mode='rgb_array')
    current_screen = get_screen(env) #env.render(mode='rgb_array')
    state = current_screen - last_screen
    
#    displayImage(state.clone().cpu().numpy())
    
    for t in count():
        # Select and perform an action
        action = select_action(state, env.action_space.n)
#        print('action: ', action.item())
        obs, reward, done, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        total_episode_reward += reward

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen(env) #env.render(mode='rgb_array')

        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
#                episode_durations.append(t + 1)
#                plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    print('Complete')
    env.render()
#    env.close()
#    plt.ioff()
#    plt.show()
    return total_episode_reward.item()


def run_evaluation(env):
    while not env.done_grading():
        run_episode(env)
        env.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('environment_filename', default='./ObstacleTower/obstacletower', nargs='?')
    parser.add_argument('--docker_training', action='store_true')
    parser.set_defaults(docker_training=False)
    args = parser.parse_args()

    env = ObstacleTowerEnv(args.environment_filename, docker_training=args.docker_training, realtime_mode=False, timeout_wait = (60*60))
    
    ##################
    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 5
    
    init_screen = env.reset()
    screen_height, screen_width, _ = init_screen.shape
    
    policy_net = DQN(screen_height, screen_width, env.action_space.n).to(device)
    target_net = DQN(screen_height, screen_width, env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)
    
    ##################
    try:
        i_episode = 0
        if env.is_grading():
            episode_reward = run_evaluation(env)
        else:
            while True:     
                episode_reward = run_episode(env, i_episode, TARGET_UPDATE)
                print("Episode reward: " + str(episode_reward))
                i_episode +=1
    finally:
        env.close()
        
        
        
        
        