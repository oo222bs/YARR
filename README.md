![Logo Missing](logo.png)

**Note**: Pirate qualification not needed to use this library.

YARR is **Y**et **A**nother **R**obotics and **R**einforcement learning framework for PyTorch.

The framework allows for asynchronous training (i.e. agent and learner running in separate processes), which makes it suitable for robot learning.
For an example of how to use this framework, see my [Attention-driven Robot Manipulation (ARM) repo](https://github.com/stepjam/ARM).

This project is mostly intended for my personal use (Stephen James) and facilitate my research.

## Modifications

This is my (Ozan Özdemir) fork of YARR for XBiT based on Mohit Shridhar's fork. Here are my modifcations:

- _independent_env_runner.py: Made it compatible for RL fine-tuning with different RL algorithms such as REINFORCE, DDPG and PPO
- rollout_generator.py: Made it compatible for RL fine-tuning with different RL algorithms such as REINFORCE, DDPG and PPO

## Install

Ensure you have [PyTorch installed](https://pytorch.org/get-started/locally/).
Then simply run:
```bash
python setup.py develop
```
