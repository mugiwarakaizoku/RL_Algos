#!/usr/bin/env python
# coding: utf-8




import gym
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
from collections import deque
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

env = gym.make('CartPole-v1')

BATCH_SIZE=32
GAMMA=0.98
LEARNING_RATE=0.005
UPDATE_TARGET_WEIGHTS_AFTER=30
REPLAY_MEMORY_SIZE=50000
TOTAL_EPISODES =5000 

total_actions = env.action_space.n
input_size = len(env.observation_space.sample())


class Replay_Memory():
    def __init__(self):
        self.buffer = deque(maxlen=REPLAY_MEMORY_SIZE)
        
    def add_transition(self,transition):
        self.buffer.append(transition)
        
    def sample(self):
        batch_size = min(BATCH_SIZE, len(self.buffer))
        batch = random.sample(self.buffer,batch_size)
        state_list,action_list,reward_list,next_state_list,done_list = [],[],[],[],[]
        for transition in batch:
            state_list.append(transition[0])
            action_list.append(transition[1])
            reward_list.append(transition[2])
            next_state_list.append(transition[3])
            done_list.append(transition[4])
        return np.array(state_list),action_list,reward_list,np.array(next_state_list),done_list
    
    def size(self):
        return len(self.buffer)


def make_NN_network(input_size,total_actions):
    model = Sequential()
    model.add(Dense(64,input_dim=input_size,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(total_actions,activation='linear'))
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model


class DQN_Agent():
    def __init__(self):
        self.q_net = make_NN_network(input_size,total_actions)
        self.target_net = make_NN_network(input_size,total_actions)
    
    def chose_action(self,state,epsilon):
        if np.random.random()>epsilon:
            state_tensor = tf.convert_to_tensor(state[None,:])
            action_tensor = self.q_net(state_tensor)
            action = np.argmax(action_tensor.numpy()[0],axis=0)
            return action
        else:
            return np.random.randint(0,total_actions)
        
        
    def train(self,mini_batch):
        state_batch, action_batch, reward_batch, next_state_batch,done_batch = mini_batch
        q_current_state = self.q_net(state_batch)
        q_val_target = np.copy(q_current_state)
        q_next_state = self.target_net(next_state_batch)
        q_next_state_max = np.max(q_next_state,axis=1)
        for i in range(state_batch.shape[0]):
            q_val_target[i][action_batch[i]] = reward_batch[i]
            if not done_batch:
                q_val_target[i][action_batch[i]] += GAMMA*q_next_state_max
        model_fit = self.q_net.fit(state_batch,q_val_target)
        return model_fit.history['loss']
    
    def update_target_network_weights(self):
         self.target_net.set_weights(self.q_net.get_weights ())



agent = DQN_Agent()
memory = Replay_Memory()
score_list = []
score = 0
for n_ep in range(TOTAL_EPISODES):
    epsilon = max(0.01,0.08-0.01*(n_ep/200))
    curr_state = env.reset()
    done=False
    while not done:
        action = agent.chose_action(curr_state,epsilon)
        next_state,reward,done,_=env.step(action)
        score+=reward
        memory.add_transition((curr_state,action,reward,next_state,done))
        curr_state = next_state
    score_list.append(score)
    sample_batch = memory.sample()
    loss = agent.train(sample_batch)
    if n_ep%UPDATE_TARGET_WEIGHTS_AFTER == 0:
        agent.update_target_network_weights()
        print('num_eps: {} loss: {} score: {}'.format(n_ep,loss,score/UPDATE_TARGET_WEIGHTS_AFTER))
        score=0
env.close()


plt.plot(score_list)
plt.ylabel('Avg_Score')
plt.xlabel('Episode')
plt.show()






