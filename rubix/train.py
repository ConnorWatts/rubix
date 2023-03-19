import sys, os

sys.path.append(os.path.abspath("."))

from rubix.config.core import config
from rubix.environment.cube.env import Cube
from rubix import utils
from rubix.agents.agent import Agent

import numpy as np
import jax


def main() -> None:
    """
    Main function for loading environment
    and training the agent.
    """

    ## random seeds & PRNG keys ##

    random_state = np.random.RandomState(config.model_config.seed)
    rng_key = jax.random.PRNGKey(
        random_state.randint(-sys.maxsize - 1, sys.maxsize + 1, dtype=np.int64)
    )

    ## load environment ##

    env = Cube(
        dim=config.cube_config.cube_dim,
        reset_steps=config.cube_config.num_moves_reset,
    )

    ##

    agent = get_agent(config.agent_config, config.network_config, env)

    ## train and eval loop


def get_agent(model_config: dict, network_config: dict, env) -> Agent:
    if model_config.agent == "DQN":
        from rubix.agents.dqn.agent import DQN
        return DQN(model_config, network_config, env)

    else:
        raise NotImplementedError("Agent {} not recognised.".format(model_config.agent))


if __name__ == "__main__":
    main()
