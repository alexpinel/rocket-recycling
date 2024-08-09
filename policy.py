import random
import numpy as np
import torch
import utils
import torch.optim as optim
from torch.distributions import Categorical

import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PositionalMapping(nn.Module):
    def __init__(self, input_dim, L=5, scale=1.0):
        super(PositionalMapping, self).__init__()
        self.L = L
        self.output_dim = input_dim * (L*2 + 1)
        self.scale = scale

    def forward(self, x):
        x = x * self.scale
        if self.L == 0:
            return x
        h = [x]
        PI = 3.1415927410125732
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x)
            x_cos = torch.cos(2**i * PI * x)
            h.append(x_sin)
            h.append(x_cos)
        return torch.cat(h, dim=-1) / self.scale

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mapping = PositionalMapping(input_dim=input_dim, L=7)
        h_dim = 128
        self.linear1 = nn.Linear(in_features=self.mapping.output_dim, out_features=h_dim, bias=True)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear3 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear4 = nn.Linear(in_features=h_dim, out_features=output_dim, bias=True)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.mapping(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x

def calculate_returns(rewards, masks, gamma=0.99):
    returns = []
    R = 0
    for r, mask in zip(reversed(rewards), reversed(masks)):
        R = r + gamma * R * mask
        returns.insert(0, R)
    return returns


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, state):
        return self.actor(state), self.critic(state)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs, _ = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, states, actions, rewards, log_probs, gamma=0.99):
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        returns = self.compute_returns(rewards, gamma)
        
        _, state_values = self.forward(states)
        state_values = state_values.squeeze()
        
        advantages = returns - state_values.detach()
        
        probs, _ = self.forward(states)
        dist = Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        
        actor_loss = -(new_log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    @staticmethod
    def compute_returns(rewards, gamma=0.99):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.FloatTensor(returns).to(device)