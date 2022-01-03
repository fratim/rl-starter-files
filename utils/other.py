import random
import numpy
import torch
import collections
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

configuration_tuple = namedtuple('configuration', 'agent_pos agent_dir box_strength goal_pos')


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

def get_values(env, agent, goal_pos, box_strength):

    n_directions = 4
    values = np.ones((env.height, env.width, n_directions)) * np.nan

    reset_config = configuration_tuple

    #if env.grid.get(*goal_pos) is not None and not env.grid.get(*goal_pos).can_overlap():
    #    return None

    for agent_x in range(env.width):
        for agent_y in range(env.height):

            # TODO remove this hack
            # if agent_x == goal_pos[0]:
            #     continue

            #if env.grid.get(x, y) is not None and not env.grid.get(x, y).can_overlap():
            #    continue

            for agent_dir in range(n_directions):

                reset_config.agent_pos = (agent_x, agent_y)
                reset_config.agent_dir = agent_dir
                reset_config.box_strength = box_strength
                reset_config.goal_pos = goal_pos

                try:
                    env.reset(configuration=reset_config)
                except:
                    continue

                env.render('human')

                obs = env.gen_obs()
                values[agent_y, agent_x, agent_dir] = agent.get_value(obs)

    return values


def plot_max_direction(env, values):
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

def plt_show_values(env, values):
    for x in range(env.width):
        for y in range(env.height):
            max_value_str = "{:.2f}".format(values[y, x])

            plt.text(x, y + 0.3, max_value_str, ha='center', va='center')


def save_valuemap(env, agent, goal_pos, box_strength, model_name):

    values = get_values(env, agent, goal_pos, box_strength)
    max_values = np.max(values, axis=2)

    plt.clf()

    # plot maximum values for each cell
    plt.imshow(max_values)

    # plot direction of maximum value for each cell
    plot_max_direction(env, values)
    plt_show_values(env, max_values)


    # plot goal position
    plt.plot(goal_pos[0], goal_pos[1] - 0.3, marker="o", color="red")

    # save to file
    model_dir = utils.get_model_dir(model_name)
    plt.savefig(os.path.join(model_dir, f"values_goalx{goal_pos[0]}y{goal_pos[1]}_box{box_strength}.png"))

    return values


