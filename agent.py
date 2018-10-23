

class Agent:
    def __init__(self, num_agents, state_size, action_size, actor, critic):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.actor = actor
        self.critic = critic

    def soft_update(self, target, src):
        pass


    def actor_forward(self):
        pass

    def critic_forward(self):
        pass
