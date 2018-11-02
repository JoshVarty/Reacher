from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
from agent import Agent
from model import ActorCriticNetwork

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



def a2c(agent, num_agents, num_episodes=400):

    all_scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, num_episodes + 1):

        avg_score = agent.step()
        scores_window.append(avg_score)
        all_scores.append(avg_score)

        if i_episode % 25 == 0:
            print("Episode score:", np.mean(scores_window), "at:", i_episode)

        if np.mean(scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.network.state_dict(), 'solution.ckpt')
            break

    return all_scores


#create agent and run against environment
agent = Agent(env, brain_name, num_agents, state_size, action_size)
scores = a2c(agent, num_agents)
            
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
