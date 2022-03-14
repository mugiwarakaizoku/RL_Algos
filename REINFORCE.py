# -*- coding: utf-8 -*-
"""REINFORCE_CartPole.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1golEkHfbmDAlChYnNjwte6eKpl8pUxL4
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

import gym
import matplotlib.pyplot as plt 
import numpy as np
import keras

from collections import deque
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp

env = gym.make('CartPole-v1')

GAMMA=0.98
LEARNING_RATE=0.001
TOTAL_EPISODES =5000

total_actions = env.action_space.n
input_size = len(env.observation_space.sample())

class REINFORCE_Network(keras.Model):
    def __init__(self):
        super(REINFORCE_Network,self).__init__()
        self.layer_1_dims=64
        self.layer_2_dims=32
        self.fc1=Dense(64,input_dim=input_size,activation='relu')
        self.fc2 = Dense(32,activation='relu')
        self.pi = Dense(total_actions,activation='softmax')
    def call(self,state):
        state_tensor = tf.convert_to_tensor([state])
        #print('state_tensor:{} and len_state_tensor:{}'.format(state_tensor,len(state_tensor)))
        value = self.fc1(state_tensor)
        value = self.fc2(value)
        policy_fn = self.pi(value)
        policy_fn = policy_fn[0]
        #print('state:{} and state_length:{}'.format(state,len(state)))
        #print('policy_fn:{} and length_of_policy_fn:{}'.format(policy_fn,len(policy_fn)))
        return policy_fn

class REINFORCE_Agent():
    def __init__(self):
        self.reinforce_network = REINFORCE_Network()
        self.reinforce_network.compile(optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
        self.data =[]

    def chose_action(self,state):
        #print('state at chose_action: {}'.format(state))
        action_probs = self.reinforce_network.call(state)
        action_prob_dist = tfp.distributions.Categorical(probs = action_probs)
        action = action_prob_dist.sample()
        action = action.numpy()
        #print(action)
        return action

    def train(self,states,actions,rewards):
        R=0
        expected_rewards=[]
        rewards.reverse()
        for i in rewards:
          R += GAMMA*i
          expected_rewards.append(R)
        expected_rewards.reverse()
        for state,action,expected_reward in zip(states,actions,expected_rewards):
          with tf.GradientTape() as tape:
            pi = self.reinforce_network.call(state)
            dist = tfp.distributions.Categorical(probs = pi)
            log_probs = dist.log_prob(action)
            loss = -log_probs*expected_reward
            gradient = tape.gradient(loss,self.reinforce_network.trainable_variables)
            self.reinforce_network.optimizer.apply_gradients(zip(gradient,self.reinforce_network.trainable_variables))
        return loss

agent = REINFORCE_Agent()
score_list = []
done = False
score = 0
for n_ep in range(TOTAL_EPISODES):
    states = []
    actions = []
    rewards = []
    curr_state = env.reset()
    done=False
    while not done:
        action = agent.chose_action(curr_state)
        next_state,reward,done,_=env.step(action)
        states.append(curr_state)
        rewards.append(reward)
        actions.append(action)
        score+=reward
        curr_state = next_state
    loss = agent.train(states,actions,rewards)
    score_list.append(score)
    if n_ep%20 == 0:
        print('num_eps: {} loss: {} score: {}'.format(n_ep,loss,score/20))
        score=0
env.close()

plt.plot(score_list)
plt.ylabel('Avg_Score')
plt.xlabel('Episode')
plt.show()
