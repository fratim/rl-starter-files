import argparse
import numpy

import utils
from utils import device
import copy
from scipy.special import softmax


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
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

def get_argmax_action(env, agent):

    all_actions = env.actions # this should return a list of actions (encoded as integers)

    initial_configuration = copy.deepcopy(env.get_current_configuration())
    values = numpy.ones((1, len(all_actions)))*numpy.nan

    for action in all_actions:
        _ = env.reset(configuration=initial_configuration)
        env.step(action.value)

        if (env.agent_pos == env.goal_pos).all():
            next_value = 1
        else:
            next_obs = env.gen_obs()
            next_value = agent.get_value(next_obs)

        values[0, action.value] = next_value

    action = numpy.random.choice(numpy.arange(0, len(all_actions)), p=softmax(30*values).T[:,0])

    _ = env.reset(configuration=initial_configuration)

    return action

for episode in range(args.episodes):
    obs = env.reset()

    print("new episode")
    max_steps = 20

    step = 0
    while True:
        env.render('human')
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

        action = get_argmax_action(env, agent)
        print(action)

        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)

        step += 1
        if done or env.window.closed or step == max_steps:
            break

    env.render_blank_image()

    if env.window.closed:
        break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
