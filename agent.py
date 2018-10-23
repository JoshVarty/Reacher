import torch

class Agent:
    def __init__(self, num_agents, state_size, action_size, actor, critic):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.actor = actor
        self.critic = critic
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def soft_update(self, target, src):
        pass


    def actor_forward(self, states):
        states = torch.tensor(states).float().to(self.device)
        logits = self.actor(states)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()

        log_prob = distribution.log_prob(action).unsqueeze(-1)
        entropy = distribution.entropy().unsqueeze(-1)
        
        return action, log_prob, entropy

    def critic_forward(self, states):
        states = torch.tensor(states).float().to(self.device)
        return None
