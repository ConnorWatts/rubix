import sys, os

sys.path.append(os.path.abspath("."))

from rubix.agents.agent import Agent
from rubix.networks import dqn_networks
import haiku as hk


class DQN(Agent):
    """Deep Q-Network agent."""

    def __init__(self, model_config, network_config, env):
        
        self.env = env
        num_actions = env.action_spec().num_values
        obsv_size = env.observation_spec()._size

        network_fn = dqn_networks.dqn_network(obsv_size, num_actions)
        self.network = hk.transform(network_fn)
