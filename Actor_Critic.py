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
        #print('state: {}'.format(state))
        state_tensor = tf.convert_to_tensor([state])
        #print('state_tensor:{} and len_state_tensor:{}'.format(state_tensor,len(state_tensor)))
        value = self.fc1(state_tensor)
        value = self.fc2(value)
        value_fn = self.v(value)[0]
        #print('state:{} and value_fn:{}'.format(state,value_fn))
        policy_fn = self.pi(value)
        policy_fn = policy_fn[0]
        #print('value_fn:{} and length_of_value_fn:{}'.format(value_fn,len(value_fn)))
        #print('state:{} and state_length:{}'.format(state,len(state)))
        #print('policy_fn:{} and length_of_policy_fn:{}'.format(policy_fn,len(policy_fn)))
        return value_fn,policy_fn

class Agent():
    def __init__(self):
        self.actor_critic = Actor_Critic_Network()
        self.actor_critic.compile(optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
        self.data =[]
    
    def add_transition(self,transition):
      self.data.append(transition)
      #print('length of self.data at add_transition: {} and self.data: {}'.format(len(self.data),self.data))

    def make_batch(self):
      state_list,action_list,reward_list,next_state_list,done_list = [],[],[],[],[]
      for transition in self.data:
        #print('length of transition at make_batch: {} and transition: {}'.format(len(transition),transition))
        state_list.append(transition[0])
        action_list.append(transition[1])
        reward_list.append(transition[2])
        next_state_list.append(transition[3])
        done_list.append(transition[4])
      self.data=[]
      #print('length of state_list at make_batch: {} and state_list: {}'.format(len(state_list),state_list))
      return np.array(state_list),action_list,reward_list,np.array(next_state_list),done_list

    def chose_action(self,state):
        #print('state at chose_action: {}'.format(state))
        _,action_probs = self.actor_critic.call(state)
        action_prob_dist = tfp.distributions.Categorical(probs = action_probs)
        # print('action_prob_dist: {}'.format(action_prob_dist))
        action = action_prob_dist.sample()
        action = action.numpy()
        print(action)
        return action

    def train(self):
        state,action,reward,next_state,done = self.make_batch()
        with tf.GradientTape() as tape:
          v_curr_state,pi_curr_state = self.actor_critic.call(state)
          #print('shape is: {}  and  state: {}'.format(tf.shape(state),state))
          #print('shape is: {}  and  v_curr_state before squeezing: {}'.format(tf.shape(v_curr_state),v_curr_state))
          v_next_state,_ = self.actor_critic.call(next_state)
          #print('state at v_next_state: {}'.format(state))
          v_curr_state = tf.squeeze(v_curr_state)
          #print('shape is: {} and  v_curr_state after squeezing: {}'.format(tf.shape(v_curr_state),v_curr_state))
          v_next_state = tf.squeeze(v_next_state)
          #print('v_curr_state: {} and curr_state_length:{}'.format(v_curr_state,len(v_curr_state)))
          #print('v_next_state: {} and next_state_length:{}'.format(v_next_state,len(v_next_state)))
          delta = reward[0]+v_next_state*GAMMA*(1-int(done[0]))-v_curr_state
          #print('reward: {} len_reward: {} v_next_state: {} v_curr_state: {}'.format(reward,len(reward),v_next_state,v_curr_state))
          #print('delta_length: {} delta: {}'.format(len(delta),delta))
          #print(type(delta))
          #print('pi_curr_state: {}'.format(pi_curr_state))
          curr_action_dist = tfp.distributions.Categorical(probs = pi_curr_state)
          #print('curr_action_dist: {}'.format(curr_action_dist))
          log_probs = curr_action_dist.log_prob(action)
          #print('log_probs: {}'.format(log_probs))

          actor_loss = -log_probs*delta
          critic_loss = delta**2
          #print('actor_loss: {}  critic_loss: {}'.format(actor_loss,critic_loss))
          #print('actor_loss_length: {} critic_loss_length: {}'.format(len(actor_loss),len(critic_loss)))
          total_loss = actor_loss + critic_loss
          total_loss = tf.reduce_sum(total_loss)
          #print(total_loss)
          #print(self.actor_critic.trainable_variables)
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

