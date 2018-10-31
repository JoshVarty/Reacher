from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from agent import Agent


env = UnityEnvironment(file_name='Reacher_Linux_NoVis/Reacher.x86_64', worker_id=12)
#env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64', worker_id=122)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# size of each state
state_size = brain.vector_observation_space_size
print('Size of each state:', state_size)


def run_until_done(env, brain_name, agent):
    online_rewards = np.zeros(num_agents)
    env_info = env.reset(train_mode=True)[brain_name]    # reset environment
    states = env_info.vector_observations                # get current state

    while True:

        states = torch.Tensor(states).cuda()
        actions, _, _ = agent.network(states)                                         # choose actions
        
        env_info = env.step(actions.cpu().detach().numpy())[brain_name]                            # send actions to environment
        next_states = env_info.vector_observations                          # get next state (for each agent)
        rewards = env_info.rewards                                          # get reward (for each agent)
        dones = env_info.local_done                                         # see if episode finished

        online_rewards += rewards
        if np.any(dones):
            break

        #Advance to next state
        states = next_states

    return np.mean(online_rewards)

def a2c(agent, num_agents, num_episodes=300):

    all_scores = []
    max_so_far = 0
    scores_window = deque(maxlen=100)

    for i_episode in range(1, num_episodes + 1):

        agent.step()
        avg_scores = run_until_done(env, brain_name, agent)
        scores_window.append(avg_scores)
        all_scores.append(avg_scores)

        avg_score = np.mean(scores_window)
        if avg_score > max_so_far:
            torch.save(agent.network.state_dict(), "partial.ckpt")
            max_so_far = avg_score

        if i_episode % 5 == 0:
            print("avg score:", avg_score, "at:", i_episode)

        if np.mean(scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.network.state_dict(), 'completed.pth')
            break

    return all_scores


#build networks
agent = Agent(env, brain_name, num_agents, state_size, action_size)
scores = a2c(agent, num_agents)
            
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
