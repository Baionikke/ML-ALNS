# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from solver import *
from instance.instance import Instance
from simulator.simulator import Simulator

ran_seed = 42
np.random.seed(ran_seed)


if __name__ == "__main__":
    current_dir = os.getcwd()

    with open(os.path.join(current_dir, "etc", "settings.json"), "r") as json_in:
        configs = json.load(json_in)

    with open(os.path.join(current_dir, "etc", "settings_solver.json"), "r") as json_in:
        configs_solver = json.load(json_in)

    configs["main_path"] = current_dir

    simulator = Simulator(configs)
    
    instance = Instance(configs, simulator)

    # configs_solver["number_iterations"] = 3
    solver = Solver(configs_solver, instance)
    solution = solver.solve(ran_seed)
    # solution.plot_soc_history()
    
    # Generi 10000 scenari: cosa succede al secondo stadio
