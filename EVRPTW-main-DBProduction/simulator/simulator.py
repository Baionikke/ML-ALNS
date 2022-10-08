# -*- coding: utf-8 -*-
import numpy as np
import random

class Simulator:
    def __init__(self, configs: dict):
        self.configs = configs
        self.get_simulation_params()
        

    def get_simulation_params(self):
        characters = self.configs["instance_file_name"][:2]
        if not characters[1].isdigit():
            self.service_generation_params = self.configs[
                self.configs["service_time_generation_type"] + "_service_time"][characters.upper()
            ]
        else:
            self.service_generation_params = self.configs[
                self.configs["service_time_generation_type"] + "_service_time"][characters[0].upper()
            ]

    def generate_service_time(self, customers):
        if self.configs["service_time_generation_type"] == "basic":
            for i in range(len(customers)):
                customers[i]["ServiceTime"] = int(
                    np.random.triangular(
                        self.service_generation_params["low"],
                        customers[i]["ServiceTime"],
                        self.service_generation_params["high"]
                    )
                )

    def generate_utilization_level(self) -> float:
        """Generate utilization level proportional to average waiting time

        Returns:
            float: utilization level
        """
        return random.uniform(self.configs["rho_low"], self.configs["rho_high"])

    # TODO: aggiungere con prob ... tempo di attesa è 0.
