import math

import numpy as np
import torch
from copy import deepcopy

from src.data.utils_data import load_cmd
from src.model.make_model import tensor_to_robot_ctrls

def simulate(robot, control_tensor):
    robot_ctrls = tensor_to_robot_ctrls(control_tensor)
    load_cmd(robot, robot_ctrls)

    x = robot.initial_state
    traj = x.detach().numpy()
    for layer in robot.layers:
        x = layer(x)
        traj = np.vstack((traj, x.detach().numpy()))

    return traj