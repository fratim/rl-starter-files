import random
import numpy
import torch
import collections
import numpy as np
import matplotlib.pyplot as plt
import os
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d

def save_valuemap(env, agent, goal_pos, model_name):

    n_directions = 4
    values = np.ones((env.height, env.width, n_directions)) * np.nan

    if env.grid.get(*goal_pos) is not None and not env.grid.get(*goal_pos).can_overlap():
        return None

    for x in range(env.width):
        for y in range(env.height):

            if env.grid.get(x, y) is not None and not env.grid.get(x, y).can_overlap():
                continue

            for agent_dir in range(n_directions):
                env.reset(agent_pos=(x, y), agent_dir=agent_dir, goal_pos=goal_pos)
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
            plt.text(x, y + 0.3, max_value_str, ha='center', va='center')

    plt.plot(goal_pos[0], goal_pos[1] - 0.3, marker="o", color="red")
    model_dir = utils.get_model_dir(model_name)
    plt.savefig(os.path.join(model_dir, f"values_goalx{goal_pos[0]}y{goal_pos[1]}.png"))
