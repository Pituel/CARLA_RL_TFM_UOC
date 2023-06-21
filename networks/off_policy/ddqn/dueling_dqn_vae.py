
import os
import torch
import torch.nn as nn
import torch.optim as optim
from settings import DQN_LEARNING_RATE, DQN_CHECKPOINT_DIR_VAE

import numpy as np
import torch.autograd as autograd


class DuelingDQnetworkVAE(nn.Module):
    def __init__(self, n_actions, model):
        # Initialize variables
        super(DuelingDQnetworkVAE, self).__init__()
        self.input_shape = (95 + 5,)
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(DQN_CHECKPOINT_DIR_VAE, model)
        self.action_space = np.arange(self.n_actions)

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        # Create the network
        self.Linear1 = nn.Sequential(
            nn.Linear(95 + 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        if torch.cuda.is_available():
            self.Linear1.cuda()
        
        self.fc_layer_inputs = self.feature_size()

        self.V = nn.Sequential(
            nn.Linear(self.fc_layer_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.A = nn.Sequential(
            nn.Linear(self.fc_layer_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions)
        )

        if self.device == 'cuda':
            self.V.cuda()
            self.A.cuda()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=DQN_LEARNING_RATE)


    def forward(self, x):
        """
        Calculate Q values

        Args:
            x: state as tensor.

        Returns:
            Q value
        """
        fc = self.Linear1(x)
        V = self.V(fc)
        A = self.A(fc)

        return V + A - A.mean()
    
    def get_action(self, state, epsilon=0.05):
        """
        Get the action to take.

        Args:
            state: current state.
            epsilon: value of epsilon.

        Returns:
            action
        """
        if np.random.random() < epsilon:
            # Random action
            action = np.random.choice(self.action_space) 
        else:
            # Action from Q-Value calculation.
            qvals = self.get_qvals(state)
            action= torch.max(qvals, dim=-1)[1].item()
        return action
    
    
    def get_qvals(self, state):
        """
        Call forward() to calculate Q-Value

        Args:
            state: current state.

        Returns:
            Q-Value
        """
        if type(state) is tuple:
            state = np.array(state)
        
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.forward(state_t)
    

    def feature_size(self):
        """
        Input size afther linear network.
        """
        return self.Linear1(autograd.Variable( torch.zeros(1, * self.input_shape)).to(device=self.device)).view(1, -1).size(1)


    def save_checkpoint(self):
        """
        Save the model
        """
        torch.save(self.state_dict(), self.checkpoint_file)


    def load_checkpoint(self):
        """
        Load the model
        """
        self.load_state_dict(torch.load(self.checkpoint_file))


