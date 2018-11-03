# Report

## Learning Algorithm

My agent uses [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) to solve the Reacher task. The bulk of the work takes place in [`Agent.step()`](https://github.com/JoshVarty/Reacher/blob/master/agent.py#L109-L120) which consists of three main steps:
1. The agent runs an episode game to completion collecting 20 rollouts
2. The agent processes the rollouts in reverse order, computing advantages
3. The agent conducts 10 optimization passes over these rollouts

### Creating Rollouts

This task is somewhat unique compared to other learning tasks. Reacher episodes are of fixed length (1,000 steps) and all 20 agents conclude at the same time. 

At each time step the agent uses its `actor` network to generate `actions` and `log_probs` for those actions. The agent also uses its `critic` network to generate `values` predictions for the current state. After taking the suggested action, the agent receives `rewards`, `dones` and `next_states` from the environment. The agent records the resulting tuple of `(states, values, actions, log_probs, rewards, dones)` into a rollout list.

### Processing Rollouts

After a rollout has been generated, the agent iterates through the rollout in reverse order while calculating `returns` and `advantages`. Discounted returns are calculated via:

`returns = rewards + self.discount_rate * dones * returns`

The `advantages` are calculated using generalized advantage estimation using `td_error`:

`td_error = rewards + self.discount_rate * dones * next_value - value`
`advantages = advantages * self.tau * self.discount_rate * dones + td_error`

In this implementation I have chosen to use a discount rate of `0.99`.

After these values are calculated, we create a tuple of `(states, actions, log_probs, returns, advantages)` and store it in a processed rollout list.

### Training the network

After processing the rollouts the agent is ready to be trained. The agent starts by shuffling the data to avoid local correlations during training. The agent then samples `64` random indices and retrieves `sampled_states`, `sampled_actions`, `sampled_log_probs_old`, `sampled_returns` and `sampled_advantages` from the processed rollout. The agent runs the policy against `(sampled_states, sampled_actions)` in order to calculate current `log_probs` and `values`. 

After generating the required values, the agent can begin calculating losses.

The loss function for PPO (without an entropy term) is given as:
![](https://i.imgur.com/aIs0oV8.png)

L<sub>t</sub><sup>Clip</sup> is given as: 

![](https://i.imgur.com/P0CjgZs.png)

```Python
ratio = (log_probs - sampled_log_probs_old).exp()
obj = ratio * sampled_advantages
obj_clipped = ratio.clamp(1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * sampled_advantages
policy_loss = -torch.min(obj, obj_clipped).mean(0)
```

L<sub>t</sub><sup>VF</sup> is given as:

![](https://i.imgur.com/dmdd8Ap.png)

```Python
value_loss = 0.5 * (sampled_returns - values).pow(2).mean()
```

After calculating `policy_loss` and `value_loss` we simply add the two losses together and backpropagate. We repeat this process `5` times in total before returning to begin collecting a new rollout.

**Note**: *I have used the [`Batcher`](https://github.com/ShangtongZhang/DeepRL/blob/95ac4ea17e82fe166b8c8b737da87db0b2097898/deep_rl/utils/misc.py#L56-L81) class taken from Shangtong Zhang's DeepRL network. I have corrected a bug in this implementation to ensure that `Batcher.reset()` is called whenever `Batch.shuffle()` is called to shuffle the data. Without this change, the agent will run one pass of optimization over the dataset, but skip the remaining passes because the batcher incorrectly reports that it has already processed all of the data.*

## Model Architecture

The actor network has the following structure:
 - A fully connected layer with `33` inputs and `256` outputs using `ReLU` activations
 - A fully connected layer with `256` inputs and `256` outputs using `ReLU` activations
 - A fully connected layer with `256` inputs and `4` outputs using `tanh` activations

The critic network has the following structure:
 - A fully connected layer with `33` inputs and `256` outputs using `ReLU` activations
 - A fully connected layer with `256` inputs and `256` outputs using `ReLU` activations
 - A fully connected layer with `256` inputs and `1` output

The `input` to both networks is the current state (a vector with `33` elements). The output of the actor represents the mean of the probability distribution for each of `4` actions. The output of the critic represents a given value for the input state.

## Hyperparameters

- Discount Rate: `0.99`
    - The degree to which we discount future rewards. This agent uses a relatively high value to encourge actions that lead to long term rewards.
    
- Tau: `0.95`
    - Weighting value used in Generalized Advantage Estimation. 

- PPO Clip: `0.02`
    - The degree to which we clip (or clamp) the probability ratio. This agent simply uses the value that was recommended in the PPO paper. 

- Learning Rate: `2e-4`
    - The rate at which we take steps in the direction of the estimated gradient. This value was chosen based on experimenting with other learning rates in the range `1e-2` to `1e-5`.

- Learning Rounds: `10`
    - The number of times we run gradient descent against a rollout generated from our policy before collecting a new rollout. 

- Minibatch size: `64`
    - The number of samples to include in an optimization batch. 

- Max Timesteps: `1000` (implicit)
    - The number of timesteps to run the environment while building a rollout. The Reacher environment is "Done" after 1,000 steps so I simply used that length to determine my rollout time.

- Gradient Clip: `5` 
    - Parameter used to limit the magnitude of the gradient and hopefully prevent unnaturally large (exploding) gradients from occurring. 

## Results

In my experience the agent can achieve an average score of 30 after ~100 episodes of training.

![]()

A sample run generated from [`Visualization.ipynb`](https://github.com/JoshVarty/Reacher/blob/master/Visualization.ipynb)

![](https://i.imgur.com/ynawSiY.gif)


## Future Work

- Investigate different length rollouts. While reading the [paper on PPO](https://arxiv.org/pdf/1707.06347.pdf) I noticed they described the length of their rollouts `T` as "much much less than the episode length". It's possible I was making my rollouts longer than they really needed to be. It would be useful to investigate whether or not learning can be achieved with much smaller rollouts.

- Try other forms of learning including TRPO or DDPG. It's possible these other forms are better at this task. 

- Incorporate entropy into the loss function. I neglected to include entropy because we did not discuss it during class and I did not understand it when reading over the code. I could probably spend more time reading about it and seeing whether or not it noticably improved my agent's performance.

- Investigating network size. It's possible I could have used more or fewer layers or have used different sizes of layers. I did not experiment very much with this hyperparameter in my experience.

- Try larger batch sizes or more learning rounds. I'm not sure if I chose these parameters correctly, but they seemed to work reasonably well. For learning rounds it might be interesting to measure how long simulation takes vs. how long training the network takes. There is probably a tradeoff to be made between simulating new experiences/rollouts and training with existing ones multiple times.