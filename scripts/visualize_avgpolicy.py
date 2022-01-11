import argparse
import numpy
import copy
import numpy as np

import utils
from utils import device


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

# determine environment rewards fn
rewards_fname = "/Users/tim/Code/blocks/rl-starter-files/storage/box4/v2-s9-ppo_box4g9/mean_values.pickle" # TODO remove this

# Load environments
env_args = {}

env = utils.make_env(args.env, seed=args.seed, env_args=env_args)
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


def most_frequent(List):
    return max(set(List), key=List.count)


def get_avg_action(agent, obs, goal_pos):

    empty_cell = np.zeros((6,), dtype=np.uint8)
    goal_cell = obs[tuple(goal_pos)]

    base_observation = copy.deepcopy(obs)
    base_observation[tuple(goal_pos)] = empty_cell

    actions = []

    for row in range(obs.shape[0]):
        for col in range(obs.shape[1]):

            manipulated_obs = copy.deepcopy(base_observation)
            if not np.array_equal(manipulated_obs[row, col], empty_cell):
                continue

            manipulated_obs[row, col] = goal_cell

            actions.append(agent.get_action(manipulated_obs))

    return most_frequent(actions)


for episode in range(args.episodes):
    obs = env.reset()

    while True:
        env.render('human')
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

        # action = agent.get_action(obs)
        action = get_avg_action(agent, obs, env.goal_pos)
        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)

        if done or env.window.closed:
            break

    env.render_blank_image()

    if env.window.closed:
        break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
