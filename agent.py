import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from model import ActorCriticNetwork

class Agent:

    def __init__(self, environment, brain_name, num_agents, state_size, action_size):
        self.environment = environment
        self.brain_name = brain_name
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.network = ActorCriticNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), 1e-4, eps=1e-5)

        self.discount_rate = 0.99
        self.tau = 0.95


    def generate_rollout(self):
        rollout = []
        online_rewards = np.zeros(self.num_agents)

        #reset environment
        env_info = self.environment.reset(train_mode=True)[self.brain_name]
        states = env_info.vector_observations

        i = 0
        while True:

            i = i + 1
            states = torch.Tensor(states).cuda()
            actions, log_probs, values = self.network(states)
            env_info = self.environment.step(actions.cpu().detach().numpy())[self.brain_name]
            rewards = env_info.rewards
            dones = np.array(env_info.local_done)
            next_states = env_info.vector_observations

            online_rewards += rewards
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), rewards, 1 - dones])
            states = next_states

            if np.any(dones):
                break

        states = torch.Tensor(states).cuda()
        _,_,last_value = self.network(states)
        rollout.append([states, last_value, None, None, None, None])
        return rollout, last_value

    def step(self):
        rollout, last_value = self.generate_rollout()
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = torch.zeros((self.num_agents, 1)).cuda()
        returns = last_value.detach()

        for i in reversed(range(len(rollout) - 1)):
            states, values, actions, log_probs, rewards, dones = rollout[i]

            dones = torch.Tensor(dones).unsqueeze(1).cuda()
            rewards = torch.Tensor(rewards).unsqueeze(1).cuda()
            #actions = torch.Tensor(actions)
            #states = torch.Tensor(states)
            next_value = rollout[i + 1][1].cuda()

            returns = rewards + (self.discount_rate * dones * returns)

            td_error = rewards + self.discount_rate * dones * next_value.detach() - values.detach()
            advantages = advantages * self.tau * self.discount_rate * dones + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()        

        mini_batch_number = 32
        batcher = Batcher(states.size(0) // mini_batch_number, [np.arange(states.size(0))])
        for _ in range(10):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = torch.Tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                _, log_probs, values = self.network(sampled_states, sampled_actions)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - 0.2,
                                          1.0 + 0.2) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0)
                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 5)
                self.optimizer.step()


class Batcher:
    
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.num_entries = len(data[0])
        self.reset()

    def reset(self):
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size

    def end(self):
        return self.batch_start >= self.num_entries

    def next_batch(self):
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start: self.batch_end])
        self.batch_start = self.batch_end
        self.batch_end = min(self.batch_start + self.batch_size, self.num_entries)
        return batch

    def shuffle(self):
        indices = np.arange(self.num_entries)
        np.random.shuffle(indices)








