from typing import Generic

from chex import Array, PRNGKey
from rubix.environment.cube.types import Observation, State

class Cube():
    def __init__(self, dim: int = 3):
        self.dim = dim

    def __repr__(self) -> str:
        return f"Cube environment with size {self.dim}."
    
    def 