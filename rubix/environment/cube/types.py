from typing import TYPE_CHECKING, NamedTuple

import jax.random

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex
import jax.numpy as jnp
from chex import Array

## maybe put in constants?

CORE_MOVES = {
    0: {"Side": 1, "Orientation": "Vertical", "Direction": "Up"},
    1: {"Side": 1, "Orientation": "Vertical", "Direction": "Down"},
    2: {"Side": 1, "Orientation": "Horizontal", "Direction": "Left"},
    3: {"Side": 1, "Orientation": "Horizontal", "Direction": "Right"},
    4: {"Side": 2, "Orientation": "Vertical", "Direction": "Up"},
    5: {"Side": 2, "Orientation": "Vertical", "Direction": "Down"},
}


@dataclass
class Move:
    rowcol: int
    side: int
    orient: str
    dir: str


@dataclass
class State:
    """
    cube: array with current cube configuration
    num_total_moves: number of performed moves
    """

    cube: Array
    num_total_moves: jnp.int32
    key: chex.PRNGKey = jax.random.PRNGKey(0)


class Observation(NamedTuple):
    """
    cube: array with current cube configuration
    """

    cube: Array
