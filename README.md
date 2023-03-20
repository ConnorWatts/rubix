# Rubix 

Rubix is a deep reinforcement learning Rubik's Cube solver written in Jax and Haiku. 

![](https://github.com/ConnorWatts/rubix/blob/main/docs/Rubiks.gif)

## Environment

The custom environment is developed in the style of the environments in [**Jumanji**](https://github.com/instadeepai/jumanji). 

Disclaimer: Since starting this repo, InstaDeep has brought out Jumanji 0.2.0 which contains a RubiksCube environment. 

## Agents

This repo currently supports DQN, QR-DQN and a discretized PPO agent. The implementations of the DQN-based agents are inspired from the [**DQN Zoo**](https://github.com/deepmind/dqn_zoo) implementations. 


## Code Structure

TBA


## How to use

### Requirements

All dependencies can be installed using:

```
pip install -r requirements/requirements.txt
```

### Training

The specific model can then be trained by running the train file.

Below is an example with the DQN agent on a 5x5 Rubik's Cube;

```
python train.py --agent=DQN --cube_dim=5

```

## TO DO

This repo is still in the early stages - below are some things I am currently working on

1) Add more agents
2) Set up a predict capacity (user input capacity) 
3) Report best results for each agent/cubesize

## Reference