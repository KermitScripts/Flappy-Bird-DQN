import random
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque


class ReplayBuffer(object):
    """A replay buffer as commonly used for off-policy Q-Learning methods."""

    def __init__(self, capacity):
        """Initializes replay buffer with certain capacity."""
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def put(self, obs, action, reward, next_obs, terminated):
        """Put a tuple of (obs, action, rewards, next_obs, terminated) into the replay buffer.
        The max length specified by capacity should never be exceeded.
        The oldest elements inside the replay buffer should be overwritten first.
        """
        self.buffer.append((obs, action, reward, next_obs, terminated))

    def get(self, batch_size):
        """Gives batch_size samples from the replay buffer."""
        return zip(*random.sample(self.buffer, batch_size))

    def __len__(self):
        """Returns the number of tuples inside the replay buffer."""
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """The neural network used to approximate the Q-function. Should output n_actions Q-values per state."""

    def __init__(self, num_obs, num_actions):
        super(DQNNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=num_obs, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=num_actions)
        )

    def forward(self, x):
        return self.layers(x)


class DQN:
    """The DQN method."""

    def __init__(self, replay_size=10000, batch_size=32, gamma=0.99, sync_after=5, lr=0.001):
        """ Initializes the DQN method.

        Parameters
        ----------
        replay_size: int
            The size of the replay buffer.
        batch_size: int
            The number of replay buffer entries an optimization step should be performed on.
        gamma: float
            The discount factor.
        sync_after: int
            Timesteps after which the target network should be synchronized with the main network.
        lr: float
            Adam optimizer learning rate.
        """

        self.obs_dim = 4
        self.act_dim = 2
        self.replay_buffer = ReplayBuffer(replay_size)
        self.sync_after = sync_after
        self.batch_size = batch_size
        self.gamma = gamma

        # Initialize DQN network
        self.dqn_net = DQNNetwork(self.obs_dim, self.act_dim)

        self.dqn_target_net = DQNNetwork(self.obs_dim, self.act_dim)
        self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())

        self.optim_dqn = optim.Adam(self.dqn_net.parameters(), lr=lr)

    def predict(self, state, epsilon):
        """Predict the best action based on state. With probability epsilon take random action

        Returns
        -------
        int
            The action to be taken.
        """
        # Implement epsilon-greedy action selection
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)  # inserts empty first dimension for potential batch-processing
            q_value = self.dqn_net.forward(state)
            action = q_value.argmax().item()
        else:
            action = random.randrange(self.act_dim)
        return action

    def compute_msbe_loss(self):
        """Compute the MSBE loss between self.dqn_net predictions and expected Q-values.

        Returns
        -------
        float
            The MSE between Q-value prediction and expected Q-values.
        """

        obs, actions, rewards, next_obs, terminated = self.replay_buffer.get(self.batch_size)

        # Convert to Tensors and stack for easier processing -> shape (batch_size, state_dimensionality)
        obs = torch.stack([torch.Tensor(ob) for ob in obs])
        next_obs = torch.stack([torch.Tensor(next_ob) for next_ob in next_obs])
 
        # Compute q_values and next_q_values -> shape (batch_size, num_actions)
        q_values = self.dqn_net(obs)

        # Choose between target and no target network
        # next_q_values = self.dqn_target_net(next_obs)
        next_q_values = self.dqn_net(next_obs)

        # Select Q-values of actions actually taken -> shape (batch_size)
        q_values = q_values.gather(1, torch.LongTensor(actions).unsqueeze(1)).squeeze(1)
        next_q_values = next_q_values.max(1)[0]

        # The target to update our network towards
        expected_q_values = torch.Tensor(rewards) + self.gamma * (1.0 - torch.Tensor(terminated)) * next_q_values

        loss = F.mse_loss(q_values, expected_q_values)
        return loss

    def epsilon_by_timestep(self, timestep, epsilon_start=0.05, epsilon_final=0.01, frames_decay=10000):
        """Linearly decays epsilon from epsilon_start to epsilon_final in frames_decay timesteps"""
        # Implement epsilon decay function
        return max(epsilon_final, epsilon_start - (float(timestep) / float(frames_decay)) * (epsilon_start - epsilon_final))

    def save_model(self, model_path):
        """Save the DQN model to a file."""
        torch.save(self.dqn_net.state_dict(), model_path)

    def load_model(self, model_path):
        """Load a DQN model from a file."""
        self.dqn_net.load_state_dict(torch.load(model_path))

