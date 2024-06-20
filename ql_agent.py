import numpy as np
import keras
import pandas as pd
import pickle
from tensorflow.keras.layers import Dense
from keras.src.optimizers import Adam
from keras.api.models import load_model
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, n_categories, action_space, batch_size, is_eval=False, model_name=''):
        '''
        state_size: number of variables
        n_categories: number of categories within Category variable
        action_size: number of actions
        batch_size: batch to train Neural Network
        is_eval: True if ANN exists
        model_name: name of model
        '''
        # For QLearning
        self.action_space = action_space
        self.num_states = n_categories
        self.q_table = np.zeros((self.num_states, len(action_space)))
        # Q-Table updating
        self.gamma = 0.95  # Disccount factor
        self.epsilon = 1.0  # Proability exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # For the Neural Network
        self.state_size = state_size
        self.batch_size = batch_size
        self.learning_rate = 0.001
        self.model = self._build_model() if not is_eval else self.load(model_name)
        self.memory = [] # Memory to store experience and use it for training

        self.current_state = None
        self.action = None
        self.n_exploit = 0
        self.n_explore = 0

    def reset_nums(self):
        self.n_exploit = 0
        self.n_explore = 0

    def get_explore_pct(self):
        return 100*self.n_explore/(self.n_explore+self.n_exploit)

    def act(self, state, cash, holdings):
        # Choose action
        self.current_state = state
        act_lim = self.action_space.copy()
        if cash == 0 or holdings == 0:
            not_act = 0 if cash == 0 else 2
            act_lim.pop(not_act)
        if np.random.uniform(0,1) < self.epsilon:
            self.action = np.random.choice(act_lim) # Explore
            self.n_explore += 1
            #print(f'Random selected action is: {self.action}')
        else:
            # Exploit
            feature_index = int(self.current_state.iloc[-1])
            if cash == 0 or holdings == 0:
                self.action = np.argmax(np.delete(self.q_table[feature_index,:], not_act))
                self.action += 1 if not_act==0 else 0
            else:
                self.action = np.argmax(self.q_table[feature_index, :])
            self.n_exploit += 1
            #print(f'Q-Table selected action is: {self.action}')
        return self.action

    def update_qtable(self, action, reward, next_state):
        # Update Q-table based on the observed reward
        if self.action is not None:
            feature_index = int(self.current_state.iloc[-1])
            # print(f'Current feature index: {feature_index}')
            next_feature_index = int(next_state.iloc[-1])
            # print(f'Next feature index: {next_feature_index}')
            current_q_value = self.q_table[feature_index, self.action]
            new_q_value = (1 - self.learning_rate) * current_q_value + self.learning_rate * (
                        reward + self.gamma * np.max(self.q_table[next_feature_index,:]))
            self.q_table[feature_index, self.action] = new_q_value

        # Update current state and action
        self.current_state = None
        self.current_action = action

    # This function is call in every episode of the training. It decreases the probability of exploration on each step
    def step(self, state, action, done):
        self.remember(state, action)
        # if len(self.memory) > self.batch_size:
           # self.replay(self.batch_size)
        if done:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    # This function will record every state and action for the training of a DNN (memory model for reuse)
    def remember(self, state, action):
        # Store in memory
        state = state.to_list()
        state.append(action)
        self.memory.append(state)

    # ARTIFICIAL NEURAL NETWORK METHODS
    def _build_model(self):
        # Build neuronal network
        model = keras.models.Sequential()
        model.add(keras.Input((self.state_size,)))
        model.add(Dense(24, activation='relu', name='hidden_layer'))
        model.add(Dense(len(self.action_space), activation='softmax', name='output_layer'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.learning_rate))
        # model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # Train model with previous experience (experience replay)
    def replay(self, batch_size):
        states = np.array([inner[:-1] for inner in self.memory])
        actions = [inner[-1] for inner in self.memory]
        actions_array = np.array([[1 if i == action else 0 for i in range(0, 3)] for action in actions])
        self.model.fit(states, actions_array, batch_size=batch_size)

    def load(self, name):
        return load_model(name)

    def save(self, name):
        self.model.save(name)