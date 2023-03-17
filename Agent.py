import torch
import torch.nn as nn
import random
from DQN_torch import DQN
import numpy as np

class DQNAgent:

    def __init__(self, state_space, action_space, max_memory_size, gamma, lr,
                 exploration_max, exploration_min, exploration_decay):

        # Define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device) # Also known as Huber loss
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = lr

        self.main_network = DQN(state_space, action_space).to(self.device)
        self.target_network = DQN(state_space, action_space).to(self.device)


    def store_transistion(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        

    # epsilon-greedy policy
    def epsilon_greedy(self, state):
        if random.uniform(0,1) < self.epsilon:
            return np.random.randint(self.action_size)
        
        Q_values = self.main_network.predict(state)
        return np.argmax(Q_values[0])

    
    #train the network
    def train(self, batch_size):
        
        #sample a mini batch of transition from the replay buffer
        minibatch = random.sample(self.replay_buffer, batch_size)

        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=self.lr)


        self.optimizer.zero_grad()
            
        
        #compute the Q value using the target network
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target_Q = (reward + self.gamma * np.amax(self.target_network(next_state, action)))
            else:
                target_Q = reward
                
       
        current_Q = self.main_network(state, action)

        loss = self.l1(current_Q, target_Q)
        loss.backward()
        self.optimizer.step() 
        self.exploration_rate *= self.exploration_decay
            

        


    #update the target network weights by copying from the main network
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())