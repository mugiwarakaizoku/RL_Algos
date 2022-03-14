# -*- coding: utf-8 -*-
"""Actor_Critic_CartPole.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-c6LExwJKKt8haeLG4vg1I4IKJhTsGTw
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

class Actor_Critic_Network(keras.Model):
    def __init__(self):
        super(Actor_Critic_Network,self).__init__()
        self.layer_1_dims=64
        self.layer_2_dims=32
        self.fc1=Dense(64,input_dim=input_size,activation='relu')
        self.fc2 = Dense(32,activation='relu')
        self.v = Dense(1,activation='linear')
        self.pi = Dense(total_actions,activation='softmax')
        
    def call(self,state):
        state_tensor = tf.convert_to_tensor([state])
        value = self.fc1(state_tensor)
        value = self.fc2(value)
        value_fn = self.v(value)[0]
        policy_fn = self.pi(value)
        policy_fn = policy_fn[0]
        return value_fn,policy_fn

class Agent():
    def __init__(self):
        self.actor_critic = Actor_Critic_Network()
        self.actor_critic.compile(optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
        self.data =[]
    
    def add_transition(self,transition):
      self.data.append(transition)

    def make_batch(self):
      state_list,action_list,reward_list,next_state_list,done_list = [],[],[],[],[]
      for transition in self.data:
        state_list.append(transition[0])
        action_list.append(transition[1])
        reward_list.append(transition[2])
        next_state_list.append(transition[3])
        done_list.append(transition[4])
      self.data=[]
      return np.array(state_list),action_list,reward_list,np.array(next_state_list),done_list

    def chose_action(self,state):
        _,action_probs = self.actor_critic.call(state)
        action_prob_dist = tfp.distributions.Categorical(probs = action_probs)
        action = action_prob_dist.sample()
        action = action.numpy()
        print(action)
        return action

    def train(self):
        state,action,reward,next_state,done = self.make_batch()
        with tf.GradientTape() as tape:
          v_curr_state,pi_curr_state = self.actor_critic.call(state)
          v_next_state,_ = self.actor_critic.call(next_state)
          v_curr_state = tf.squeeze(v_curr_state)
          v_next_state = tf.squeeze(v_next_state)
          delta = reward[0]+v_next_state*GAMMA*(1-int(done[0]))-v_curr_state
          curr_action_dist = tfp.distributions.Categorical(probs = pi_curr_state)
          log_probs = curr_action_dist.log_prob(action)

          actor_loss = -log_probs*delta
          critic_loss = delta**2
          total_loss = actor_loss + critic_loss
          total_loss = tf.reduce_sum(total_loss)
          gradient = tape.gradient(total_loss,self.actor_critic.trainable_variables)
          self.actor_critic.optimizer.apply_gradients(zip(gradient,self.actor_critic.trainable_variables))
        return total_loss

"""### Model Training"""

agent = Agent()
score_list = []
score = 0
for n_ep in range(TOTAL_EPISODES):
    curr_state = env.reset()
    done=False
    while not done:
        action = agent.chose_action(curr_state)
        next_state,reward,done,_=env.step(action)
        score+=reward
        agent.add_transition((curr_state,action,reward,next_state,done))
        curr_state = next_state
    score_list.append(score)
    loss = agent.train()
    if n_ep%20 == 0:
        print('num_eps: {} loss: {} score: {}'.format(n_ep,loss,score/20))
        score=0
env.close()

plt.plot(score_list)
plt.ylabel('Avg_Score')
plt.xlabel('Episode')
plt.show()

