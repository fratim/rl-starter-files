import argparse
import numpy

import utils
from utils import device

import numpy as np
import matplotlib.pyplot as plt
import pickle

import os


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, seed=args.seed)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")

# Run the agent

if args.gif:
   from array2gif import write_gif
   frames = []

# Create a window to view the environment
env.render('human')

pb_goals = []
for goal_x in range(env.width):
    for goal_y in range(env.height):
        pb_goals.append((goal_x, goal_y))

max_box_strength = env.max_box_strength
mean_values = np.ones((env.height, env.width, max_box_strength+1))*np.nan

for box_strength in range(max_box_strength+1):

    env.reset()

    values = np.ones((env.height, env.width, len(pb_goals)))*np.nan

    for goal_id, goal_pos in enumerate(pb_goals):
        values_in = utils.get_values(env, agent, goal_pos, box_strength)
        max_values = np.max(values_in, axis=2)
        values[:, :, goal_id] = max_values

    values_mean_slice = np.nanmean(values, axis=2)
    mean_values[:, :, box_strength] = values_mean_slice

    plt.clf()
    plt.imshow(values_mean_slice)
    utils.plt_show_values(env, values_mean_slice)

    model_dir = utils.get_model_dir(args.model)
    plt.savefig(os.path.join(model_dir, f"values_avg_box{box_strength}.png"))

output_fname = os.path.join(model_dir, "mean_values.pickle")
with open(output_fname, "wb") as output_file:
    pickle.dump(mean_values, output_file)

# for goal_x in range(env.width):
#     for goal_y in range(env.height):
#         for box_strength in range(5):
#             utils.save_valuemap(env, agent, (goal_x, goal_y), box_strength, args.model)
