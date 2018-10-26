import torch

class Agent:
    def __init__(self, num_agents, state_size, action_size, actor, critic):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.actor = actor
        self.critic = critic
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.std_dev = torch.nn.Parameter(torch.ones(num_agents, action_size)).to(self.device)

    def soft_update(self, target, src):
        pass


    def actor_forward(self, states):
        states = torch.tensor(states).float().to(self.device)
        means = self.actor(states)
        distributions = torch.distributions.Normal(means, self.std_dev)
        action = distributions.sample()
        log_prob = distributions.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        
        return action, log_prob

    def critic_forward(self, states):
        states = torch.tensor(states).float().to(self.device)
        return None
