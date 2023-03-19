import functools
from typing import Optional, Tuple
from chex import Array, PRNGKey
from jax import random

import sys, os

sys.path.append(os.path.abspath("."))
from rubix.environment.types import (
    TimeStep,
    restart,
    termination,
    transition,
    truncation,
)
from rubix.environment.cube.types import State, Observation
from rubix.environment.cube.utils import generate_cube


class Cube:
    def __init__(self, dim: int, reset_steps: int):
        self.dim = dim
        self.reset_steps = reset_steps

    def __repr__(self) -> str:
        return f"Cube environment with size {self.dim}."

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep[Array]]:
        """Resets the environment.
        Args:
            key: random key used to reset the cube to a random state.
        Returns:
            state: State object corresponding to the new state of the environment.
            timestep: TimeStep object corresponding the first timestep returned by the environment.
        """
        move_key, reset_key = random.split(key)
        cube = generate_cube(self.dim, self.reset_steps, move_key, reset_key)

    def step(self, state: State, action: Array) -> Tuple[State, TimeStep[Array]]:
        """Perform an environment step.
        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the actions to take.
                - 0 no op
                - 1 move left
                - 2 move up
                - 3 move right
                - 4 move down
        Returns:
            state: State object corresponding to the next state of the environment,
            timestep: TimeStep object corresponding the timestep returned by the environment,
        """
