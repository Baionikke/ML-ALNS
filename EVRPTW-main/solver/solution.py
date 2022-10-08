# -*- coding: utf-8 -*-
from __future__ import annotations
import copy
import numpy as np
import random
from instance import Instance
import matplotlib.pyplot as plt


class Solution:
    def __init__(self, instance: Instance):
        self.routes = []
        self.routes_with_stations = []
        self.vehicles = []
        self.arrival_times = []
        self.instance = instance
        self.station_info = []
    
    # TODO EDO: add_node non potrebbe essere add_recharging_station, etc? così risparmiamo setup_node_attributes?
    # penso di no perchè setup node attributes serve per fargli scoprire che tipo di nodo è
    # e a monte della funzione non lo sa
    def add_node(self, node : str, route_idx : int, position_idx : int):
        self.routes[route_idx].insert(position_idx, self.setup_node_attributes(node))
        # self.update_station_info(route_idx, position_idx, self.routes[route_idx][position_idx], removed = False)
        self.update_route(route_idx)

    def remove_node(self, node : dict, route_idx : int):
        # self.update_station_info(route_idx, self.routes[route_idx].index(node), node, removed = True)
        self.routes[route_idx].remove(node)
        self.update_route(route_idx)

    def remove_route(self, route_idx):
        self.routes.pop(route_idx)
        self.vehicles.pop(route_idx)
        self.arrival_times.pop(route_idx)
        self.station_info.pop(route_idx)

    # TODO: unire
    def add_fresh_route(self, starting_time=None):
        self.routes.append([self.setup_node_attributes("D0")])
        self.vehicles.append(copy.deepcopy(dict(self.instance.base_electric_vehicle)))
        if starting_time:
            self.vehicles[-1]["time"] = starting_time
        self.arrival_times.append([])
        self.station_info.append([])

    def add_route_time(self, starting_time):
        self.routes.append([self.setup_node_attributes("D0")])
        self.vehicles.append(copy.deepcopy(dict(self.instance.base_electric_vehicle)))
        self.vehicles[-1]["time"] = starting_time
        self.arrival_times.append([])
        self.station_info.append([])

    def add_built_route(self, route, vehicle, arrival_time, station_info):
        self.routes.append(route)
        self.vehicles.append(vehicle)
        self.arrival_times.append(arrival_time)
        self.station_info.append(station_info)

    def copy_solution(self, solution : Solution):
        # provare a usare il metodo alternativo di at dataclass
        solution_copy = Solution(solution.instance)
        for j, route in enumerate(solution.routes):
            solution_copy.add_fresh_route()
            for i in range(1, len(route)):
                solution_copy.add_node(route[i]["name"], j, i)

        return solution_copy

    def generate_route_id(self):
        routes = []
        for i, route in enumerate(self.routes):
            routes.append([])
            for node in route:
                routes[i].append(node["name"])
        return routes

    def update_station_waiting_time(self, route_idx, old_waiting_time):
        # this method will update the station waiting time and recalculate the times
        if self.station_info[route_idx] != []:
            station = self.instance.stations_dict[self.routes[route_idx][self.station_info[route_idx][0]]["name"]]
            experiment = random.random()
            if experiment < station["utilization_level"]:
                new_waiting_time = 0
            else:
                new_waiting_time = np.random.exponential(1 / self.instance.mu)
            # i need the station in study
            for station_position in self.station_info[route_idx]:
                for i in range(station_position, len(self.routes[route_idx])):
                    self.arrival_times[route_idx][i - 1] += old_waiting_time - new_waiting_time
            
    def generate_station_info_route(self, route_idx: int):
        self.station_info[route_idx] = []
        for i, node in enumerate(self.routes[route_idx]):
            if node["isStation"]:
                self.station_info[route_idx].append(i)

    # def update_station_info(self, route_idx : int, position_idx : int, node : dict, removed = False):
    #     new_station = False
    #     if len(self.station_info[route_idx]) == 0:
    #         # if new node is station and station info empty, append it
    #         if node["isStation"]:
    #             self.station_info[route_idx].append(position_idx)
    #     else:
    #         if removed:
    #             # if we are removing a node
    #             if node["isStation"]:
    #                 self.station_info[route_idx].remove(position_idx)
    #             for i in range(len(self.station_info[route_idx])):
    #                 if self.station_info[route_idx][i] > position_idx:
    #                     self.station_info[route_idx][i] -= 1
    #         else:
    #             # if we are adding a node and the list is not empty
    #             if node["isStation"]: # if it's a station add it
    #                 self.station_info[route_idx].append(position_idx)
    #                 new_station = True
    #             # for every element in station info, if the new node has < index than station, update station index
    #             for i in range(len(self.station_info[route_idx])):
    #                 if self.station_info[route_idx][i] > position_idx:
    #                     # if position_idx is before a station
    #                     self.station_info[route_idx][i] += 1
    #                 elif self.station_info[route_idx][i] == position_idx:
    #                     if not new_station:
    #                         self.station_info[route_idx][i] += 1

    def remove_empty_routes(self):
        found = False
        for i, route in enumerate(self.routes):
            for node in route:
                if len(route) == 2 and node["isDepot"]:
                    empty_route = i
                    found = True
                elif len(route) == 3 and node["isStation"]:
                    empty_route = i
                    found = True
            if found:
                self.remove_route(empty_route)
                found = False
                if empty_route in self.routes_with_stations:
                    self.routes_with_stations.remove(empty_route)

    def update_route(self, route_idx, partial = False, target_SoC = 0.9):
        self.vehicles[route_idx] = copy.deepcopy(dict(self.instance.base_electric_vehicle))
        self.arrival_times[route_idx] = []

        for i in range(1, len(self.routes[route_idx])):
            if self.routes[route_idx][i]["isCustomer"]:
                self.update_ev_customer_pickup(route_idx, i)
            elif self.routes[route_idx][i]["isStation"]:
                if partial:
                    self.update_ev_station_partial(route_idx, i, target_SoC)
                else:
                    self.update_ev_station(route_idx, i)
                if route_idx not in self.routes_with_stations:
                    self.routes_with_stations.append(route_idx)
            elif self.routes[route_idx][i]["isDepot"]:
                self.update_ev_depot(route_idx, i)

    def setup_node_attributes(self, node):
        attributes = {}
        attributes.update({"name" : node})
        if "C" in node:
            attributes.update({"isCustomer" : True, "isStation" : False, "isDepot" : False})
        elif "S" in node:
            attributes.update({"isCustomer" : False, "isStation" : True, "isDepot" : False})
        else:
            attributes.update({"isCustomer" : False, "isStation" : False, "isDepot" : True})
        return attributes

    def update_ev_customer_pickup(self, route_idx, position_idx):
        customer = self.instance.customers_dict[self.routes[route_idx][position_idx]["name"]]
        distance_travelled = self.instance.distance_matrix[self.routes[route_idx][position_idx-1]["name"]][self.routes[route_idx][position_idx]["name"]]
        self.vehicles[route_idx]["SoC"] = (self.vehicles[route_idx]["SoC"]*self.instance.Q - self.instance.h * distance_travelled) / self.instance.Q
        self.vehicles[route_idx]["SoC_list"].append(self.vehicles[route_idx]["SoC"])
        self.vehicles[route_idx]["current_cargo"] += customer["demand"]
        if (distance_travelled/self.instance.average_velocity + self.vehicles[route_idx]["time"]) < customer["ReadyTime"]:
            self.arrival_times[route_idx].append(customer["ReadyTime"] + self.vehicles[route_idx]["time"])
            self.vehicles[route_idx]["time"] += customer["ReadyTime"] + customer["ServiceTime"]
        else:
            self.arrival_times[route_idx].append(distance_travelled/self.instance.average_velocity + self.vehicles[route_idx]["time"])
            self.vehicles[route_idx]["time"] += customer["ServiceTime"] + distance_travelled/self.instance.average_velocity

    def update_ev_station(self, route_idx, position_idx):
        distance_travelled = self.instance.distance_matrix[self.routes[route_idx][position_idx-1]["name"]][self.routes[route_idx][position_idx]["name"]]
        self.vehicles[route_idx]["SoC"] = (self.vehicles[route_idx]["SoC"]*self.instance.Q - self.instance.h * distance_travelled) / self.instance.Q
        self.vehicles[route_idx]["SoC_list"].append(self.vehicles[route_idx]["SoC"])
        expected_waiting_time = self.instance.get_expected_waiting_time_v2(self.instance.stations_dict[self.routes[route_idx][position_idx]["name"]]["utilization_level"])
        past_SoC = self.vehicles[route_idx]["SoC"]
        self.vehicles[route_idx]["SoC"] += 0.9 - past_SoC
        # TODO EDO: perché qui sotto expected_waiting_time?
        self.vehicles[route_idx]["time"] += expected_waiting_time + (self.instance.E_recharge/4 *( 0.9 - past_SoC ) * 10)
        self.arrival_times[route_idx].append(distance_travelled/self.instance.average_velocity + self.vehicles[route_idx]["time"])

    def update_ev_station_partial(self, route_idx, position_idx, target_SoC):
        distance_travelled = self.instance.distance_matrix[self.routes[route_idx][position_idx-1]["name"]][self.routes[route_idx][position_idx]["name"]]
        self.vehicles[route_idx]["SoC"] = (self.vehicles[route_idx]["SoC"]*self.instance.Q - self.instance.h * distance_travelled) / self.instance.Q
        self.vehicles[route_idx]["SoC_list"].append(self.vehicles[route_idx]["SoC"])
        expected_waiting_time = self.instance.get_expected_waiting_time_v2(self.instance.stations_dict[self.routes[route_idx][position_idx]["name"]]["utilization_level"])
        past_SoC = self.vehicles[route_idx]["SoC"]
        self.vehicles[route_idx]["SoC"] += target_SoC - past_SoC
        self.vehicles[route_idx]["time"] += expected_waiting_time + (self.instance.E_recharge/4 *( target_SoC - past_SoC ) * 10)
        self.arrival_times[route_idx].append(distance_travelled/self.instance.average_velocity + self.vehicles[route_idx]["time"])

    def update_ev_depot(self, route_idx, position_idx):
        distance_travelled = self.instance.distance_matrix[self.routes[route_idx][position_idx-1]["name"]][self.routes[route_idx][position_idx]["name"]]
        self.vehicles[route_idx]["SoC"] = (self.vehicles[route_idx]["SoC"]*self.instance.Q - self.instance.h * distance_travelled) / self.instance.Q
        self.vehicles[route_idx]["SoC_list"].append(self.vehicles[route_idx]["SoC"])
        self.vehicles[route_idx]["time"] += distance_travelled / self.instance.average_velocity
        self.arrival_times[route_idx].append(distance_travelled/self.instance.average_velocity + self.vehicles[route_idx]["time"])

    def plot_soc_history(self):
        for i, ele in enumerate(self.vehicles):
            plt.plot(ele["SoC_list"], label=f"Vehicle {i}")
            plt.xlabel('pick up')
            plt.ylabel('SoC')

    # TODO EDO: rimuovere?
    '''
    arrival_times = []
    def (waiting):
        sum( arrival + waiting - due)
        
        [
            S []
            S [.... X .... ]
            []
        ]
    '''