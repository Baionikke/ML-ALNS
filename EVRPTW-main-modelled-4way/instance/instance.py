# -*- coding: utf-8 -*-
import os
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from simulator import *

class Instance:
    def __init__(self, configs: dict, simulator: Simulator):
        self.simulator = simulator
        self.cost_driver = configs["driver_wage"]
        self.cost_overtime = configs["overtime_cost_numerator"] / configs["overtime_cost_denominator"]
        self.cost_vehicle = configs["fixed_vehicle_acquisition"]
        self.cost_energy = configs["unit_energy_cost"]
        self.rho_low = configs["rho_low"] # TODO: spiegare cosa sono
        self.rho_high = configs["rho_high"]
        self.Q = 0.0 # battery tank capacity
        self.C = 0.0 # vehicle cargo capacity
        self.h = 0.0 # vehicle consumption rate
        self.g = 0.0 # vehicle recharging rate
        self.average_velocity = 0.0

        self.customers = []
        # self.electric_vehicles = []
        self.charging_stations = []
        self.start_point = {} # TODO: rename depot_point
        self.distance_matrix = pd.DataFrame()
        self.base_electric_vehicle = {}

        self.setup_from_file(
            configs["main_path"],
            configs["instance_file_name"]
        )
        self.generate_distance_matrix()
        self.initial_setup_electric_vehicle()
        self.setup_utilization_level()
        self.simulator.generate_service_time(self.customers)
        self.setup_customers_dict()
        self.setup_stations_dict()

        self.E_recharge = 0.4 * self.g * self.Q
        self.mu = 1/self.E_recharge

    def setup_from_file(self, main_path, instance_file_name): # TODO: rename read_from_solomon_instance
        file_in = open(
            os.path.join(
                main_path, "data",
                instance_file_name),
            "r"
        )
        lines = file_in.readlines()
        file_in.close()

        header = lines.pop(0).strip("\n").split()
        # header.append(header[-1][-11:])
        # header[-2] = header[-2][:-11]
        
        base_dict = {
            "StringID" : "",
            "Type" : "",
            "x" : "",
            "y" : "",
            "demand" : "",
            "ReadyTime" : "",
            "DueDate" : "",
            "ServiceTime" : ""
        }
        # for field in header:
        #     base_dict.update({field : ""})

        
        line = lines.pop(0).strip("\n")
        while line != "":
            line = line.split()
            tmp_dict = dict(base_dict)
            for i, key in enumerate(tmp_dict.keys()):
                if key != "StringID" and key != "Type":
                    tmp_dict[key] = float(line[i])
                else:
                    tmp_dict[key] = line[i]

            if tmp_dict["Type"] == "c":
                self.customers.append(tmp_dict)
            elif tmp_dict["Type"] == "f":
                tmp_dict["utilization_level"] = 0.0
                self.charging_stations.append(tmp_dict)
            else:
                self.start_point = dict(tmp_dict)

            line = lines.pop(0).strip("\n")

        self.Q = float(lines.pop(0).strip("\n").split("/")[-2])
        self.C = float(lines.pop(0).strip("\n").split("/")[-2])
        self.h = float(lines.pop(0).strip("\n").split("/")[-2])
        self.g = float(lines.pop(0).strip("\n").split("/")[-2])
        self.average_velocity = float(lines.pop(0).strip("\n").split("/")[-2])
        
    def setup_customers_dict(self):
        self.customers_dict = {}
        for customer in self.customers:
            self.customers_dict.update({
                customer["StringID"] : {}
            })
            for field in customer:
                if field != "StringID":
                    self.customers_dict[customer["StringID"]].update({
                        field : customer[field]
                    })

    def setup_stations_dict(self):
        self.stations_dict = {}
        for station in self.charging_stations:
            self.stations_dict.update({
                station["StringID"] : {}
            })
            for field in station:
                if field != "StringID":
                    self.stations_dict[station["StringID"]].update({
                        field : station[field]
                    })

    def generate_distance_matrix(self):
        row_dict = {"StringID" : "", "x" : "", "y" : ""}
        row_dict["StringID"] = [self.start_point["StringID"]]
        row_dict["x"] = [self.start_point["x"]]
        row_dict["y"] = [self.start_point["y"]]

        for station in self.charging_stations:
            row_dict["StringID"].append(station["StringID"])
            row_dict["x"].append(station["x"])
            row_dict["y"].append(station["y"])

        for customer in self.customers:
            row_dict["StringID"].append(customer["StringID"])
            row_dict["x"].append(customer["x"])
            row_dict["y"].append(customer["y"])

        tmp_df = pd.DataFrame(data = row_dict, columns = row_dict.keys())

        self.distance_matrix = pd.DataFrame(
            squareform(pdist(tmp_df.iloc[:, 1:])),
            columns=tmp_df.StringID.unique(),
            index=tmp_df.StringID.unique()
        )

    def initial_setup_electric_vehicle(self):
        self.base_electric_vehicle["SoC"] = 1
        self.base_electric_vehicle["current_cargo"] = 0
        self.base_electric_vehicle["time"] = 0
        self.base_electric_vehicle["depot_closure"] = self.start_point["DueDate"]
        self.base_electric_vehicle["SoC_list"] = [1]

    def get_expected_waiting_time(self, idx_station):
        rho = self.charging_stations[idx_station]["utilization_level"]
        lambd = self.mu * rho
        return rho/(self.mu-lambd)

    def get_expected_waiting_time_v2(self, utilization_level):
        rho = utilization_level
        lambd = self.mu * rho
        return rho/(self.mu - lambd)
    # TODO: chi viene usato? get_expected_waiting_time o get_expected_waiting_time_v2, meglio get_expected_waiting_time
    # TODO: pulire dai v2 il codice

    def setup_utilization_level(self):
        for i in range(len(self.charging_stations)):
            self.charging_stations[i]["utilization_level"] = self.simulator.generate_utilization_level()

    def compute_OF(self, solution): # TODO: spostare in Solution e togliere argomento solution
        of = 0
        
        distance = 0 # distance travelled cost 
        tot_time = 0 # driver paid for the time cost 
        overtime = 0 # driver overtime pay 
        fixed_cost = 0 # fixed vehicle acquisition cost 
        for route in solution.routes:
            for i in range(len(route)-1):
                distance += self.distance_matrix[route[i]["name"]][route[i+1]["name"]]
        
        for arrival_time in solution.arrival_times:
            overtime = arrival_time[-1] - self.start_point["DueDate"]
            if overtime <= 0:
                tot_time += arrival_time[-1]
            else:
                overtime += overtime
        
        fixed_cost = len(solution.routes)
        of = (self.cost_energy * distance) + (self.cost_driver * tot_time)
        of += (self.cost_overtime * overtime) + (self.cost_vehicle * fixed_cost)

        return of 

    def compute_cost_route(self, route, arrival_time): # TODO: spostare in Solution
        cost = 0
        
        distance = 0 # distance travelled cost 
        tot_time = 0 # driver paid for the time cost 
        overtime = 0 # driver overtime pay 
        fixed_cost = 0 # fixed vehicle acquisition cost 

        for i in range(len(route)-1):
            distance += self.distance_matrix[route[i]["name"]][route[i+1]["name"]]

        overtime = arrival_time[-1] - self.start_point["DueDate"]
        if overtime <= 0:
            tot_time += arrival_time[-1]
        else:
            overtime += overtime
        
        fixed_cost = len(route)
        cost = (self.cost_energy * distance) + (self.cost_driver * tot_time)
        cost += (self.cost_overtime * overtime) + (self.cost_vehicle * fixed_cost)

        return cost


    def compute_arc_cost(self, node1, node2):
        cost = 0
        distance = self.distance_matrix[node1][node2]
        travel_time = distance/self.average_velocity
        cost += (distance * self.cost_energy)
        cost += (travel_time * self.cost_driver)
        return cost

    def compute_energy_cost(self, route):
        route_distance = 0
        for i in range(len(route)-1):
            route_distance += self.distance_matrix[route[i]["name"]][route[i+1]["name"]]
        route_energy = self.cost_energy * route_distance

        return route_energy

