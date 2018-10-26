import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNetwork(nn.Module):

    def __init__(self, state_size, output_size, activation_function=None):
        super(FCNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.activation_function = activation_function

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.activation_function is not None:
            x = self.activation_function(x)

        return x

class ActorCriticNetwork(nn.Module):

    def __init__(self, state_size, action_size, device):
        super(ActorCriticNetwork, self).__init__()

        self.actor = FCNetwork(state_size, action_size, torch.tanh)
        self.critic = FCNetwork(state_size, 1)
        self.device = device

        self.std = nn.Parameter(torch.ones(1, action_size)).to(self.device)
        self.to(self.device)


    def forward(self, state):
        state = torch.Tensor(state).to(self.device)

        a = self.actor(state)
        distribution = torch.distributions.Normal(a, self.std)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        value = self.critic(state)

        return action, log_prob, value

