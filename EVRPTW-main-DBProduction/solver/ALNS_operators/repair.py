# -*- coding: utf-8 -*-
import abc
from abc import abstractmethod
import numpy as np

from solver.solution import Solution


class Repair(object):
    """Base class for repair operators"""

    __metaclass__ = abc.ABCMeta

    @abstractmethod
    def __init__(self, setting):
        self.setting = setting

    @abstractmethod
    def apply(self, solution : Solution, removed_elements):
        pass

    def __str__(self):
        return self.name

class GreedyRepairCustomer(Repair):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "GreedyRepairCustomer"

    def apply(self, solution : Solution, removed_elements):
        base_cost = self.setting["instance"].compute_OF(solution)
        num_iterations = len(removed_elements)
        removed = [False] * len(removed_elements)

        for _ in range(num_iterations):
            best_spot = {}
            max_cost_difference = -np.Inf
            insertion_costs = []
            for c, customer_to_insert in enumerate(removed_elements):
                insertion_costs.append([])
                if not removed[c]:
                    for i, route in enumerate(solution.routes):
                        for j in range(len(route)):
                            if j != 0:
                                solution.add_node(customer_to_insert, i, j)
                                
                                cost_difference = base_cost - self.setting["instance"].compute_OF(solution)
                                if cost_difference > max_cost_difference:
                                    max_cost_difference = cost_difference
                                    best_spot["route"] = i
                                    best_spot["position"] = j
                                    insertion_idx = customer_to_insert

                                solution.remove_node(route[j], i)
            solution.add_node(insertion_idx, best_spot["route"], best_spot["position"])
            removed[removed_elements.index(insertion_idx)] = True


class ProbabilisticGreedyRepairCustomer(Repair):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "ProbabilisticGreedyRepairCustomer"

    def apply(self, solution: Solution, removed_elements):
        base_cost = self.setting["instance"].compute_OF(solution)
        removed = [False] * len(removed_elements)

        for _ in range(len(removed_elements)):
            best_spot = {}
            max_cost_difference = - np.Inf
            insertion_costs = []
            for c, customer_to_insert in enumerate(removed_elements):
                insertion_costs.append([])
                if not removed[c]:
                    for i, route in enumerate(solution.routes):
                        for j in range(1, len(route)):
                            
                            solution.add_node(customer_to_insert, i, j)
                            recourse_cost = self.setting["recourse_cost"](solution, i)
                            
                            cost_difference = base_cost - (self.setting["instance"].compute_OF(solution) + recourse_cost)
                            if cost_difference > max_cost_difference:
                                max_cost_difference = cost_difference
                                best_spot["route"] = i
                                best_spot["position"] = j
                                insertion_idx = customer_to_insert

                            solution.remove_node(route[j], i)


            solution.add_node(insertion_idx, best_spot["route"], best_spot["position"])
            removed[removed_elements.index(insertion_idx)] = True


class ProbabilisticGreedyConfidenceRepairCustomer(Repair):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "ProbabilisticGreedyConfidenceRepairCustomer"

    def apply(self, solution : Solution, removed_elements: list):
        base_cost = self.setting["instance"].compute_OF(solution)
        num_iterations = len(removed_elements)
        removed = [False] * len(removed_elements)
        
        for _ in range(num_iterations):
            insertion_costs = []
            insertion_failsafe = []
            best_spot = {}
            max_cost_difference = -1000000
            for c, customer_to_insert in enumerate(removed_elements):
                insertion_costs.append([])
                insertion_failsafe.append([])
                if not removed[c]:
                    for i, route in enumerate(solution.routes):
                        for j in range(1, len(route)):     
                            solution.add_node(customer_to_insert, i, j)
                            tmp = base_cost - self.setting["instance"].compute_OF(solution)

                            insertion_failsafe[c].append(
                                {
                                    "spot" : {"route" : i,"position" : j}, 
                                    "cost_difference" : tmp
                                }
                            )
                            if tmp > max_cost_difference:
                                max_cost_difference = tmp
                                best_spot["route"] = i
                                best_spot["position"] = j
                                insertion_idx = customer_to_insert

                            feasible_prob = self.setting["route_feasible"](solution, i) 
                            if feasible_prob > self.setting["configs"]["greedy_confidence"]:
                                insertion_costs[c].append(
                                    {
                                        "spot" : {"route" : i,"position" : j}, 
                                        "cost_difference" : tmp
                                    }
                                )
                                if tmp > max_cost_difference:
                                    max_cost_difference = tmp
                                    best_spot["route"] = i
                                    best_spot["position"] = j
                                    insertion_idx = customer_to_insert
                                

                            solution.remove_node(route[j], i)

                        if insertion_costs[c] == []:
                            insertion_failsafe[c].append(False)

            count = 0
            for c, spot in enumerate(insertion_failsafe):
                if spot != []:
                    if spot[-1] == False:
                        count += 1
                if removed[c]:
                    count += 1

            if count == len(insertion_costs):
                solution.add_node(insertion_idx, best_spot["route"], best_spot["position"])
                removed[removed_elements.index(insertion_idx)] = True
                    
            else:
                solution.add_node(insertion_idx, best_spot["route"], best_spot["position"])
                removed[removed_elements.index(insertion_idx)] = True


class DeterministicBestRepairStation(Repair):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "DeterministicBestRepairStation"

    def apply(self, solution : Solution, removed_elements):
        base_cost = self.setting["instance"].compute_OF(solution)
        num_iterations = len(removed_elements)
        removed = [False] * len(removed_elements)
        r = 0 # index of removed elements list

        routes = solution.routes


        for j, route in enumerate(routes):

            entered = False
            for i in range(len(route)):
                if solution.vehicles[j]["SoC_list"][i] < 0.1 and not entered and route[i]["isCustomer"]:
                    customer_pos = [j, i]
                    entered = True

            feasible_position = False
            if entered:
                while not feasible_position and not all(removed):
                    insertion_costs = []
                    for station in self.setting["instance"].charging_stations:
                        solution.add_node(
                            station["StringID"],
                            customer_pos[0], 
                            customer_pos[1]
                            )
                        insertion_costs.append(
                                        {
                                            "cost_difference" : base_cost - self.setting["instance"].compute_OF(solution),
                                            "name" : station["StringID"]
                                        }
                                    )

                        solution.remove_node(solution.routes[customer_pos[0]][customer_pos[1]], customer_pos[0])

                    if insertion_costs != []:

                        max_cost_difference = -1000000
                        for spot in insertion_costs:
                            if spot["cost_difference"] > max_cost_difference:
                                max_cost_difference = spot["cost_difference"]
                                insertion_idx = spot["name"]

                        solution.add_node(insertion_idx, customer_pos[0], customer_pos[1])

                        feasible_position = True
                        removed[r] = True
                        r += 1
                    else:
                        customer_pos[1] -= 1


class ProbabilisticBestRepairStation(Repair):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "ProbabilisticBestRepairStation"

    def apply(self, solution : Solution, removed_elements):
        base_cost = self.setting["instance"].compute_OF(solution)
        num_iterations = len(removed_elements)
        removed = [False] * len(removed_elements)
        r = 0

        routes = solution.routes

        for j, route in enumerate(routes):
            entered = False
            for i in range(len(route)):
                if solution.vehicles[j]["SoC_list"][i] < 0.1 and not entered and route[i]["isCustomer"]:
                    customer_pos = [j, i]
                    entered = True

            feasible_position = False
            if entered:
                while not feasible_position  and not all(removed):
                    insertion_costs = []
                    max_cost_difference = -1000000
                    for station in self.setting["instance"].charging_stations:
                        solution.add_node(
                            station["StringID"],
                            customer_pos[0], 
                            customer_pos[1]
                            )
                        recourse_cost = self.setting["recourse_cost"](solution, i)
                        insertion_costs.append(
                                        {
                                            "cost_difference" : base_cost - (self.setting["instance"].compute_OF(solution)+ recourse_cost),
                                            "name" : station["StringID"]
                                        }
                                    )
                        cost_difference = base_cost - (self.setting["instance"].compute_OF(solution)+ recourse_cost)
                        if cost_difference > max_cost_difference:
                            max_cost_difference = cost_difference
                            insertion_idx = station["StringID"]

                        solution.remove_node(solution.routes[customer_pos[0]][customer_pos[1]], customer_pos[0])

                    if insertion_costs != []:

                        solution.add_node(insertion_idx, customer_pos[0], customer_pos[1])

                        feasible_position = True
                        removed[r] = True
                        r += 1
                    else:
                        customer_pos[1] -= 1