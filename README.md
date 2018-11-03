# Reacher

Unity's Reacher Environment is an environment in which and agent must control robotic hands with the goal of keeping the ends of the arms within a sphere. This repository uses the 20-agent environment in which a single agent must control 20 robotic arms.

The agent interacts with the environment via the following:

  - It is fed 20 sets of observations each with a vector of 33 elements
  - For each of the 20 arms, the agent must provide an action vector representing 4 continous actions



![https://github.com/Unity-Technologies/ml-agents/blob/master/docs/images/reacher.png?raw=true](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/images/reacher.png?raw=true)

*Sample frame taken from:* https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher


This repository trains an agent to attain an average score (over 100 episodes and across 20 arms) of at least 30. It trains the agent using PPO ([Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)). This implementation of PPO was derived from: https://github.com/ShangtongZhang/DeepRL

## Prerequisites

- Anaconda
- Python 3.6
- A `conda` environment created as follows

  - Linux or Mac:
  ```
  conda create --name drlnd python=3.6
  source activate drlnd 
  ```

  - Windows
  ```
  conda create --name drlnd python=3.6 
  activate drlnd
  ```

- Required dependencies

```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

## Getting Started

1. `git clone https://github.com/JoshVarty/Reacher.git`

2. `cd Reacher`

3. Download Unity Reacher Environment
   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
   - Linux Headless: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

4. Unzip to git directory

5. `jupyter notebook`

6. You can train your own agent via [`main.ipynb`](https://github.com/JoshVarty/Reacher/blob/master/main.ipynb) or watch a single episode of the pre-trained network via [`Visualization.ipynb`](https://github.com/JoshVarty/Reacher/blob/master/Visualization.ipynb)

## Results

In my experience the agent can achieve an average score of 30 after ~100 episodes of training.

![](https://i.gyazo.com/e14d5c2b30a12a4c17af517423ed3033.png)

A sample run generated from [`Visualization.ipynb`](https://github.com/JoshVarty/Reacher/blob/master/Visualization.ipynb)

![](https://i.imgur.com/ynawSiY.gif)


## Notes
 - Only tested on Ubuntu 18.04
 - Details of the learning algorithm, architecture and hyperparameters can be found in `Report.md`