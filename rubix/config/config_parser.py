import argparse

parser = argparse.ArgumentParser(description="Rubix parameters")

# rubiks cube parameters
parser.add_argument("--cube_dim", type=int, help="Dimension of side of cube", default=3)
parser.add_argument("--num_moves_reset", type=int, help="Number of moves to scramble the cube", default=3)

args = parser.parse_args()
