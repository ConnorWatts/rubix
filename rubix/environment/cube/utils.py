from chex import Array, PRNGKey
from jax import numpy as jnp
from jax import random

from rubix.environment.cube.types import Move, CORE_MOVES


def update_cube(cube: Array, move: Move) -> Array:
    if move.orient == "Horizontal" and move.dir == "Left":
        return horizontal_left_move(cube, move.rowcol, move.side)
    elif move.orient == "Horizontal" and move.dir == "Right":
        return horizontal_right_move(cube, move.rowcol, move.side)
    elif move.orient == "Vertical" and move.dir == "Up":
        return vertical_up_move(cube, move.rowcol, move.side)
    elif move.orient == "Vertical" and move.dir == "Down":
        return vertical_down_move(cube, move.rowcol, move.side)


## generate functions ##


def parse_move(move_idx, cube_dim) -> Move:
    row_col = move_idx % cube_dim
    core_move_idx = move_idx // cube_dim
    return Move(
        rowcol=row_col,
        side=CORE_MOVES[core_move_idx]["Side"],
        orient=CORE_MOVES[core_move_idx]["Orientation"],
        dir=CORE_MOVES[core_move_idx]["Direction"],
    )


def generate_cube(
    cube_dim: jnp.int32, reset_steps: jnp.int32, move_key: PRNGKey, reset_key: PRNGKey
) -> Array:
    cube = generate_solved_cube(cube_dim=cube_dim)
    # TO DO: look into fori loop
    for move_idx in random.randint(
        move_key, shape=(reset_steps,), minval=0, maxval=cube_dim * 6
    ):
        move = parse_move(move_idx, cube_dim)
        cube = update_cube(cube, move)
    return cube


def generate_solved_cube(cube_dim: jnp.int32) -> Array:
    cvals = [1, 2, 3, 4, 5, 6]
    return jnp.array(
        [[[val for i in range(cube_dim)] for j in range(cube_dim)] for val in cvals]
    )


### movers ###


def horizontal_left_move(cube: Array, row: int, side: int) -> Array:
    return_cube = jnp.copy(cube)

    return_cube = return_cube.at[0, row].set(cube[1, row])
    return_cube = return_cube.at[1, row].set(cube[2, row])
    return_cube = return_cube.at[2, row].set(cube[3, row])
    return_cube = return_cube.at[3, row].set(cube[0, row])

    if row == 0:
        return_cube = return_cube.at[4].set(jnp.rot90(cube[4], axes=(1, 0)))
    elif row == cube[0, 0].size - 1:
        return_cube = return_cube.at[5].set(jnp.rot90(cube[5]))

    return return_cube


def horizontal_right_move(cube: Array, row: int, side: int) -> Array:
    return_cube = jnp.copy(cube)

    return_cube = return_cube.at[0, row].set(cube[3, row])
    return_cube = return_cube.at[1, row].set(cube[0, row])
    return_cube = return_cube.at[2, row].set(cube[1, row])
    return_cube = return_cube.at[3, row].set(cube[2, row])

    if row == 0:
        return_cube = return_cube.at[4].set(jnp.rot90(cube[4]))
    elif row == cube[0, 0].size - 1:
        return_cube = return_cube.at[5].set(jnp.rot90(cube[5], axes=(1, 0)))

    return return_cube


def vertical_up_move(cube: Array, col: int, side: int) -> Array:
    return_cube = jnp.copy(cube)

    if side == 1:
        return_cube = return_cube.at[0, :, col].set(cube[5][:, col])
        return_cube = return_cube.at[4, :, col].set(cube[0][:, col])
        return_cube = return_cube.at[2, :, (cube[0, 0] - 1) - col].set(
            cube[4][:, col][::-1]
        )
        return_cube = return_cube.at[5, :, col].set(
            cube[2][:, (cube[0, 0] - 1) - col][::-1]
        )
        if col == 0:
            return_cube = return_cube.at[3].set(jnp.rot90(cube[3]))
        elif col == cube[0, 0] - 1:
            return_cube = return_cube.at[1].set(jnp.rot90(cube[1], axes=(1, 0)))

    elif side == 2:
        cube = self.cube.copy()
        (
            cube[1][:, col][:],
            cube[4][self.dim - 1 - col],
            cube[3][:, (self.dim - 1) - col],
            cube[5][col],
        ) = (
            self.cube[5][col][::-1],
            self.cube[1][:, col],
            self.cube[4][self.dim - 1 - col],
            self.cube[3][:, (self.dim - 1) - col],
        )
        self.cube = cube
        if col == 0:
            self.cube[0] = np.rot90(self.cube[0])
        elif col == self.dim - 1:
            self.cube[2] = np.rot90(self.cube[2], axes=(1, 0))

    def vertical_down(self, col, side) -> None:
        cube = self.cube.copy()

        if side == 1:
            (
                cube[0][:, col],
                cube[4][:, col][:],
                cube[2][:, (self.dim - 1) - col][:],
                cube[5][:, col],
            ) = (
                self.cube[4][:, col],
                self.cube[2][:, (self.dim - 1) - col][::-1],
                self.cube[5][:, col][::-1],
                self.cube[0][:, col],
            )
            self.cube = cube
            if col == 0:
                self.cube[3] = np.rot90(self.cube[3], axes=(1, 0))
            elif col == self.dim - 1:
                self.cube[1] = np.rot90(self.cube[1])
        elif side == 2:
            (
                cube[1][:, col],
                cube[4][self.dim - 1 - col],
                cube[3][:, (self.dim - 1) - col],
                cube[5][col],
            ) = (
                self.cube[4][self.dim - 1 - col],
                self.cube[3][:, (self.dim - 1) - col],
                self.cube[5][col],
                self.cube[1][:, col],
            )
            self.cube = cube
            if col == 0:
                self.cube[0] = np.rot90(self.cube[0], axes=(1, 0))
            elif col == self.dim - 1:
                self.cube[2] = np.rot90(self.cube[2])
