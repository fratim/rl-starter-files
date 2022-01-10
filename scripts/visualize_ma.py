import argparse
import numpy

import utils
from utils import device
import gym

import time

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=10,
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
action_space_org = gym.spaces.discrete
action_space_org.n = 5
#
model_dir = utils.get_model_dir(args.model)
agent_l = utils.Agent(env.observation_space, action_space_org, model_dir, argmax=args.argmax, use_memory=args.memory, use_text=args.text)

#model_dir_a = utils.get_model_dir("v2-avg")
#agent_a = utils.Agent(env.observation_space, action_space_org, model_dir, argmax=args.argmax, use_memory=args.memory, use_text=args.text)

# Run the agent

if args.gif:
   from array2gif import write_gif
   frames = []

# Create a window to view the environment
env.render('human')

for episode in range(args.episodes):
    obs = env.reset()

    while True:
        env.render('human')
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

        #action_l = agent_l.get_action(obs[0])
        #action_a = agent_a.get_action(obs[1])
        action_l = agent_l.get_action(obs)

        #action = [numpy.random.randint(0, env.action_space.n) for _ in range(len(env.agents))]
        #action = [action_l, action_a]
        action = action_l


        obs, reward, done, _ = env.step(action)

        print(reward) if reward > 0 else None
        # agent.analyze_feedback(reward, done)

        time.sleep(0.1)

        if done or env.window.closed:
            break



    if env.window.closed:
        break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
