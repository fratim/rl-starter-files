import argparse
import numpy

import utils
from utils import device

import numpy as np
import matplotlib.pyplot as plt

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

env = utils.make_env(args.env, args.seed)
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
n_directions = 4
values = np.ones((env.height, env.width, n_directions))*np.nan

for x in range(env.width):
    for y in range(env.height):

        if env.grid.get(x, y) is not None and not env.grid.get(x, y).can_overlap():
            continue

        for agent_dir in range(n_directions):

            env.reset(agent_pos=(x, y), agent_dir=agent_dir)
            env.render('human')

            obs = env.gen_obs()
            values[y, x, agent_dir] = agent.get_value(obs)

plt.clf()
plt.imshow(np.max(values, axis=2))

for x in range(env.width):
    for y in range(env.height):
        if len(np.argwhere(values[y, x, :] == np.max(values[y, x, :]))) == 0:
            continue

        agent_dir_max = np.argwhere(values[y, x, :] == np.max(values[y, x, :]))[0][0]

        if agent_dir_max == 0:
            disp_marker = ">"
        elif agent_dir_max == 1:
            disp_marker = "v"
        elif agent_dir_max == 2:
            disp_marker = "<"
        elif agent_dir_max == 3:
            disp_marker = "^"
        else:
            raise NotImplementedError


        max_value = np.max(values[y, x, :])
        max_value_str = "{:.2f}".format(max_value)

        plt.plot(x, y, marker=disp_marker, color="black")
        plt.text(x, y+0.3, max_value_str, ha='center', va='center')


model_dir = utils.get_model_dir(args.model)
plt.savefig(os.path.join(model_dir, "values.png"))