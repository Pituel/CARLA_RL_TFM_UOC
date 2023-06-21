import os
import torch
import torch.nn as nn
import torch.optim as optim
from settings import DQN_LEARNING_RATE, DQN_CHECKPOINT_DIR_CNN

import numpy as np


class DuelingDQnetwork(nn.Module):
    def __init__(self, n_actions, model):
        # Initialize variables
        super(DuelingDQnetwork, self).__init__()
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(DQN_CHECKPOINT_DIR_CNN, model)
        self.action_space = np.arange(self.n_actions)

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        # Create the network
        self.image_features = nn.Sequential(
            nn.Conv2d(3 , 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.driving_features = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        if torch.cuda.is_available():
            self.image_features.cuda()
            self.driving_features.cuda()
        
        self.V = nn.Sequential(
            nn.Linear(6208, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.A = nn.Sequential(
            nn.Linear(6208, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions)
        )

        if self.device == 'cuda':
            self.V.cuda()
            self.A.cuda()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=DQN_LEARNING_RATE)


    def forward(self, x, y):
        """
        Calculate Q values

        Args:
            x: state as tensor.
            y: driving features as tensor.

        Returns:
            Q value
        """

        # Pepare both tensor to be concated.
        x = self.image_features(x).reshape(-1, 6144)
        yd = self.driving_features(y)
        if(len(y) == 5):
            yd = yd.unsqueeze(0)
        z = torch.cat((x, yd), dim=1)

        V = self.V(z)
        A = self.A(z)
        
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
        Prepare the array-state for the forward function.
        Call forward() to calculate Q-Value

        Args:
            state: current state as array.

        Returns:
            Q-Value
        """

        if(len(state) == 2):
            state = np.array(state, dtype=object)
        else:
            im = np.array([i[0] for i in state])
            dv = np.array([i[1] for i in state])
    
        if(len(state) == 2):
            state_i = torch.FloatTensor(state[0]).to(device=self.device)
            state_d = torch.FloatTensor(state[1]).to(device=self.device)
        else:
            state_i = torch.FloatTensor(im).to(device=self.device)
            state_d = torch.FloatTensor(dv).to(device=self.device)
        

        return self.forward(state_i, state_d)

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

