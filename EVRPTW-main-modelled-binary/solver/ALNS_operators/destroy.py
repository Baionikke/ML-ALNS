# -*- coding: utf-8 -*-
import abc
import random
import numpy as np
import copy
from abc import abstractmethod
from solver.solution import Solution


class Destroy(object):
    """Base class for destroy operators"""

    __metaclass__ = abc.ABCMeta

    @abstractmethod
    def __init__(self, setting):
        self.setting = setting

    @abstractmethod
    def apply(self, solution : Solution):
        pass

    def __str__(self):
        return self.name


class GreedyDestroyCustomer(Destroy):

    def __init__(self, setting):
        # TODO: sostituisci la riga sotto con:
        super().__init__(setting)
        self.name = "GreedyDestroyCustomer"
        # idem per gli altri operatori
        # self.setting = setting

    def apply(self, solution : Solution):
        customers_to_remove = []
        while len(customers_to_remove) < self.setting["configs"]["gamma_c"]:
            potential_removal = self.setting["instance"].customers[
                random.randrange(0, len(self.setting["instance"].customers), 1)
            ]["StringID"]
            if potential_removal not in customers_to_remove:
                customers_to_remove.append(potential_removal)

        temporary_solution = solution.copy_solution(solution)
        for i, route in enumerate(temporary_solution.routes):
            for idx in route:
                if idx["name"] in customers_to_remove:
                    solution.remove_node(idx, i)
                    
        return customers_to_remove


class WorstDistanceDestroyCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "WorstDistanceDestroyCustomer"

    def apply(self, solution : Solution):
        customers_to_remove = []

        distance_costs = {}
        for route in solution.routes:
            for j, node in enumerate(route):
                if node["isCustomer"]:
                    cost_node_1 = self.setting["instance"].compute_arc_cost(route[j-1]["name"], route[j]["name"])
                    cost_node_2 =  self.setting["instance"].compute_arc_cost(route[j]["name"], route[j+1]["name"])
                    distance_cost = abs(cost_node_1 + cost_node_2)
                    distance_costs.update({node["name"] : distance_cost})

        while len(customers_to_remove) < self.setting["configs"]["gamma_c"]:
            sorted_distance_costs = dict(sorted(distance_costs.items(), key = lambda item: item[1], reverse = True))
            lamb = random.uniform(0, 1)
            position_to_remove = np.floor((lamb**self.setting["configs"]["worst_removal_determinism_factor"]) * len(sorted_distance_costs))        
            customer_to_remove = list(distance_costs.keys())[int(position_to_remove)]
            if customer_to_remove not in customers_to_remove:
                customers_to_remove.append(customer_to_remove)

        temporary_solution = solution.copy_solution(solution)
        for i, route in enumerate(temporary_solution.routes):
            for idx in route:
                if idx["name"] in customers_to_remove:
                    solution.remove_node(idx, i)

        return customers_to_remove


class WorstTimeDestroyCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "WorstTimeDestroyCustomer"

    def apply(self, solution : Solution):
        customers_to_remove = []
        # early_times = {}
        # for customer in self.setting["instance"].customers_dict:
        #     early_times.update({customer["StringID"] : customer["ReadyTime"]})

        time_costs = {}
        for i, route in enumerate(solution.routes):
            for j, node in enumerate(route):
                if node["isCustomer"]:
                    
                    cost = np.abs(self.setting["instance"].customers_dict[node["name"]]["ReadyTime"] - solution.arrival_times[i][j])
                    time_costs.update({node["name"] : cost})

        while len(customers_to_remove) < self.setting["configs"]["gamma_c"]:
            sorted_time_costs = dict(sorted(time_costs.items(), key = lambda item: item[1], reverse = True))
            lamb = random.uniform(0, 1)
            position_to_remove = np.floor((lamb**self.setting["configs"]["worst_removal_determinism_factor"]) * len(sorted_time_costs))        
            customer_to_remove = list(time_costs.keys())[int(position_to_remove)]
            if customer_to_remove not in customers_to_remove:
                customers_to_remove.append(customer_to_remove)

        temporary_solution = solution.copy_solution(solution)
        for i, route in enumerate(temporary_solution.routes):
            for idx in route:
                if idx["name"] in customers_to_remove:
                    solution.remove_node(idx, i)

        return customers_to_remove


class ShawDestroyCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "ShawDestroyCustomer"

    def apply(self, solution : Solution):
        customers_to_remove = []
        shaw_lambda = self.setting["configs"]["shaw_parameters"]["lambda"] # distance relatedness
        shaw_mu = self.setting["configs"]["shaw_parameters"]["mu"] # time relatedness
        shaw_nu = self.setting["configs"]["shaw_parameters"]["nu"] # capacity relatedness
        shaw_csai = self.setting["configs"]["shaw_parameters"]["csai"] # possible serving vehicles relatedness

        removal = self.setting["instance"].customers[
                            random.randrange(0, len(self.setting["instance"].customers), 1)
                            ]
        customers_to_remove.append(removal["StringID"])

        routes_names = solution.generate_route_id()
        relatedness = {}
        for customer in self.setting["instance"].customers:
            shaw_terms = 0
            if customer["StringID"] not in removal["StringID"]:
                shaw_terms += self.setting["instance"].distance_matrix[removal["StringID"]][customer["StringID"]] * shaw_lambda
                shaw_terms += np.abs(customer["ReadyTime"] - removal["ReadyTime"]) * shaw_mu
                shaw_terms += np.abs(customer["demand"] - removal["demand"]) * shaw_nu
                gammaij = 1
                for route in routes_names:
                    if customer["StringID"] in route and removal["StringID"] in route:
                        gammaij = -1
                shaw_terms += shaw_csai * gammaij
                relatedness.update({customer["StringID"] : shaw_terms})

        while len(customers_to_remove) < self.setting["configs"]["gamma_c"]:
            sorted_relatedness = dict(sorted(relatedness.items(), key = lambda item: item[1], reverse = True))
            lamb = random.uniform(0, 1)
            position_to_remove = np.floor((lamb**self.setting["configs"]["shaw_removal_determinism_factor"]) * len(sorted_relatedness))        
            customer_to_remove = list(sorted_relatedness.keys())[int(position_to_remove)]
            if customer_to_remove not in customers_to_remove:
                customers_to_remove.append(customer_to_remove)

        temporary_solution = solution.copy_solution(solution)
        for i, route in enumerate(temporary_solution.routes):
            for idx in route:
                if idx["name"] in customers_to_remove:
                    solution.remove_node(idx, i)

        return customers_to_remove


class ProximityBasedDestroyCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "ProximityBasedDestroyCustomer"

    def apply(self, solution : Solution):
        customers_to_remove = []
        shaw_lambda = self.setting["configs"]["shaw_parameters"]["lambda"] # distance relatedness
        shaw_mu = 0 # time relatedness
        shaw_nu = 0 # capacity relatedness
        shaw_csai = 0 # possible serving vehicles relatedness

        removal = self.setting["instance"].customers[
                            random.randrange(0, len(self.setting["instance"].customers), 1)
                            ]
        customers_to_remove.append(removal["StringID"])

        routes_names = solution.generate_route_id()
        relatedness = {}
        for customer in self.setting["instance"].customers:
            shaw_terms = 0
            if customer["StringID"] not in removal["StringID"]:
                shaw_terms += self.setting["instance"].distance_matrix[removal["StringID"]][customer["StringID"]] * shaw_lambda
                shaw_terms += np.abs(customer["ReadyTime"] - removal["ReadyTime"]) * shaw_mu
                shaw_terms += np.abs(customer["demand"] - removal["demand"]) * shaw_nu
                gammaij = 1
                for route in routes_names:
                    if customer["StringID"] in route and removal["StringID"] in route:
                        gammaij = -1
                shaw_terms += shaw_csai * gammaij
                relatedness.update({customer["StringID"] : shaw_terms})

        while len(customers_to_remove) < self.setting["configs"]["gamma_c"]:
            sorted_relatedness = dict(sorted(relatedness.items(), key = lambda item: item[1], reverse = True))
            lamb = random.uniform(0, 1)
            position_to_remove = np.floor((lamb**self.setting["configs"]["shaw_removal_determinism_factor"]) * len(sorted_relatedness))        
            customer_to_remove = list(relatedness.keys())[int(position_to_remove)]
            if customer_to_remove not in customers_to_remove:
                customers_to_remove.append(customer_to_remove)

        temporary_solution = solution.copy_solution(solution)
        for i, route in enumerate(temporary_solution.routes):
            for idx in route:
                if idx["name"] in customers_to_remove:
                    solution.remove_node(idx, i)

        return customers_to_remove


class TimeBasedDestroyCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "TimeBasedDestroyCustomer"
        

    def apply(self, solution : Solution):
        customers_to_remove = []
        shaw_lambda = 0 # distance relatedness
        shaw_mu = self.setting["configs"]["shaw_parameters"]["mu"] # time relatedness
        shaw_nu = 0 # capacity relatedness
        shaw_csai = 0 # possible serving vehicles relatedness

        removal = self.setting["instance"].customers[
                            random.randrange(0, len(self.setting["instance"].customers), 1)
                            ]
        customers_to_remove.append(removal["StringID"])

        routes_names = solution.generate_route_id()
        relatedness = {}
        for customer in self.setting["instance"].customers:
            shaw_terms = 0
            if customer["StringID"] not in removal["StringID"]:
                shaw_terms += self.setting["instance"].distance_matrix[removal["StringID"]][customer["StringID"]] * shaw_lambda
                shaw_terms += np.abs(customer["ReadyTime"] - removal["ReadyTime"]) * shaw_mu
                shaw_terms += np.abs(customer["demand"] - removal["demand"]) * shaw_nu
                gammaij = 1
                for route in routes_names:
                    if customer["StringID"] in route and removal["StringID"] in route:
                        gammaij = -1
                shaw_terms += shaw_csai * gammaij
                relatedness.update({customer["StringID"] : shaw_terms})

        while len(customers_to_remove) < self.setting["configs"]["gamma_c"]:
            sorted_relatedness = dict(sorted(relatedness.items(), key = lambda item: item[1], reverse = True))
            lamb = random.uniform(0, 1)
            position_to_remove = np.floor((lamb**self.setting["configs"]["shaw_removal_determinism_factor"]) * len(sorted_relatedness))        
            customer_to_remove = list(relatedness.keys())[int(position_to_remove)]
            if customer_to_remove not in customers_to_remove:
                customers_to_remove.append(customer_to_remove)

        temporary_solution = solution.copy_solution(solution)
        for i, route in enumerate(temporary_solution.routes):
            for idx in route:
                if idx["name"] in customers_to_remove:
                    solution.remove_node(idx, i)

        return customers_to_remove


class DemandBasedDestroyCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "DemandBasedDestroyCustomer"

    def apply(self, solution : Solution):
        customers_to_remove = []
        shaw_lambda = 0 # distance relatedness
        shaw_mu = 0 # time relatedness
        shaw_nu = self.setting["configs"]["shaw_parameters"]["nu"] # capacity relatedness
        shaw_csai = 0 # possible serving vehicles relatedness

        removal = self.setting["instance"].customers[
                            random.randrange(0, len(self.setting["instance"].customers), 1)
                            ]
        customers_to_remove.append(removal["StringID"])

        routes_names = solution.generate_route_id()
        relatedness = {}
        for customer in self.setting["instance"].customers:
            shaw_terms = 0
            if customer["StringID"] not in removal["StringID"]:
                shaw_terms += self.setting["instance"].distance_matrix[removal["StringID"]][customer["StringID"]] * shaw_lambda
                shaw_terms += np.abs(customer["ReadyTime"] - removal["ReadyTime"]) * shaw_mu
                shaw_terms += np.abs(customer["demand"] - removal["demand"]) * shaw_nu
                gammaij = 1
                for route in routes_names:
                    if customer["StringID"] in route and removal["StringID"] in route:
                        gammaij = -1
                shaw_terms += shaw_csai * gammaij
                relatedness.update({customer["StringID"] : shaw_terms})

        while len(customers_to_remove) < self.setting["configs"]["gamma_c"]:
            sorted_relatedness = dict(sorted(relatedness.items(), key = lambda item: item[1], reverse = True))
            lamb = random.uniform(0, 1)
            position_to_remove = np.floor((lamb**self.setting["configs"]["shaw_removal_determinism_factor"]) * len(sorted_relatedness))        
            customer_to_remove = list(relatedness.keys())[int(position_to_remove)]
            if customer_to_remove not in customers_to_remove:
                customers_to_remove.append(customer_to_remove)

        temporary_solution = solution.copy_solution(solution)
        for i, route in enumerate(temporary_solution.routes):
            for idx in route:
                if idx["name"] in customers_to_remove:
                    solution.remove_node(idx, i)

        return customers_to_remove



class ZoneDestroyCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "ZoneDestroyCustomer"

    def apply(self, solution : Solution):
        customers_to_remove = []
        # find corner points
        xmin = 999
        ymin = 999
        xmax = 0
        ymax = 0
        for customer in self.setting["instance"].customers:
            if customer["x"] < xmin:
                xmin = customer["x"]
            if customer["y"] < ymin:
                ymin = customer["y"]
            if customer["x"] > xmax:
                xmax = customer["x"]
            if customer["y"] > ymax:
                ymax = customer["y"]

        edgex = np.floor(np.abs(xmin-xmax)/2)
        edgey = np.floor(np.abs(ymin-ymax)/2)
        rectangles = [
            [(edgex, ymin), (xmin , edgey), (edgex, edgey), (xmin, ymin)],
            [(xmax, ymin), (xmax, edgey), (edgex, edgey), (edgex, ymin)],
            [(xmax, edgey), (xmax, ymax), (edgex, ymax), (edgex, edgey)],
            [(edgex, edgey), (edgex, ymax), (xmin, ymax), (xmin, edgey)]
        ]
        
        which_zone = np.random.randint(0, len(rectangles))
        zone = rectangles[which_zone]

        while len(customers_to_remove) < self.setting["configs"]["gamma_c"]:
            for customer in self.setting["instance"].customers:
                inzone = customer["x"] > zone[3][0] and customer["y"] > zone[3][1] and customer["x"] < zone[1][0] and customer["y"] < zone[1][1]
                if inzone and len(customers_to_remove) < self.setting["configs"]["gamma_c"] and customer["StringID"] not in customers_to_remove:
                    customers_to_remove.append(customer["StringID"])
            
            new_which_zone = np.random.randint(0, len(rectangles))
            while new_which_zone == which_zone:
                new_which_zone = np.random.randint(0, len(rectangles))
            zone = rectangles[new_which_zone]

        temporary_solution = solution.copy_solution(solution)
        for i, route in enumerate(temporary_solution.routes):
            for idx in route:
                if idx["name"] in customers_to_remove:
                    solution.remove_node(idx, i)

        return customers_to_remove


class RandomRouteDestroyCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "RandomRouteDestroyCustomerstomer"

    def apply(self, solution : Solution):
        customers_to_remove = []
        which_route = np.random.randint(0, len(solution.routes))
        route_to_remove = solution.routes[which_route]
        loop_counter = 0
        go_in_else = False

        while len(customers_to_remove) < self.setting["configs"]["gamma_c"]:
            if route_to_remove != ["D0", "D0"] and ((len(customers_to_remove)+2) != len(route_to_remove)) and not go_in_else :
                which_customer = np.random.randint(0, len(route_to_remove))
                loop_counter += 1
                if route_to_remove[which_customer]["isCustomer"] and route_to_remove[which_customer]["name"] not in customers_to_remove:
                    customers_to_remove.append(route_to_remove[which_customer]["name"])
                if loop_counter == 20:
                    go_in_else = True
                    loop_counter = 0

            else:
                new_which_route = np.random.randint(0, len(solution.routes))
                while new_which_route == which_route:
                    new_which_route = np.random.randint(0, len(solution.routes))
                route_to_remove = solution.routes[new_which_route]
                go_in_else = False

        temporary_solution = solution.copy_solution(solution)
        for i, route in enumerate(temporary_solution.routes):
            for idx in route:
                if idx["name"] in customers_to_remove:
                    solution.remove_node(idx, i)

        return customers_to_remove


class GreedyRouteRemoval(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "GreedyRouteRemovalestroyCustomer"

    def apply(self, solution : Solution):
        customers_to_remove = []
        route_costs = {}
        for i, route in enumerate(solution.routes):
            route_cost = 0
            for j in range(len(route)-1):
                route_cost += self.setting["instance"].compute_arc_cost(route[j]["name"], route[j+1]["name"])
            route_costs.update({ str(i) : route_cost})

        sorted_costs = dict(sorted(route_costs.items(), key = lambda item: item[1], reverse = True))
        route_index = 0
        which_route = int(list(sorted_costs.keys())[route_index])

        while len(customers_to_remove) < self.setting["configs"]["gamma_c"]:
            for i, idx in enumerate(solution.routes[which_route]):
                if idx["name"] not in customers_to_remove and idx["isCustomer"] and len(customers_to_remove) < self.setting["configs"]["gamma_c"]:
                    customers_to_remove.append(idx["name"])

            if len(customers_to_remove) < self.setting["configs"]["gamma_c"]:
                route_index += 1
                which_route = int(list(sorted_costs.keys())[route_index])

        temporary_solution = solution.copy_solution(solution)
        for i, route in enumerate(temporary_solution.routes):
            for idx in route:
                if idx["name"] in customers_to_remove:
                    solution.remove_node(idx, i)

        return customers_to_remove


class ProbabilisticWorstRemovalCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "ProbabilisticWorstRemovalCustomer"

    def apply(self, solution : Solution):
        customers_to_remove = []
        stations_positions = []
        probability_infeasible_next_customer = {}

        for j, route in enumerate(solution.routes):
            for i, node in enumerate(route):
                if node["isStation"]:
                    stations_positions.append([ j, i ])
            
        for j, route in enumerate(solution.routes):
            for i, node in enumerate(route):
                for station in stations_positions:
                    if j == station[0]:
                        if i < station[1]:
                            if node["name"] not in probability_infeasible_next_customer.keys() and node["isCustomer"]:
                                probability_infeasible_next_customer.update({ node["name"] : 0 })
                        else:
                            if node["isCustomer"]:
                                probability_infeasible_next_customer.update({node["name"] : self.setting["customer_infeasible"](solution, [j, i])})
                    else:
                        if node["name"] not in probability_infeasible_next_customer.keys() and node["isCustomer"]:
                            probability_infeasible_next_customer.update({ node["name"] : 0 })
                        
        sorted_prob = dict(sorted(probability_infeasible_next_customer.items(), key = lambda item: item[1], reverse = True))
        while len(customers_to_remove) < self.setting["configs"]["gamma_c"]:
            for customer in sorted_prob:
                if customer not in customers_to_remove and len(customers_to_remove) < self.setting["configs"]["gamma_c"]:
                    customers_to_remove.append(customer)
        
        temporary_solution = solution.copy_solution(solution)
        for i, route in enumerate(temporary_solution.routes):
            for idx in route:
                if idx["name"] in customers_to_remove:
                    solution.remove_node(idx, i)

        return customers_to_remove


class RandomDestroyStation(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "RandomDestroyStationtroyCustomer"

    def apply(self, solution : Solution):
        stations_to_remove = []
        possible_stations = []

        for route in solution.routes:
            for node in route:
                if node["isStation"]:
                    possible_stations.append(node["name"])

        if possible_stations != []:
            if len(possible_stations) > self.setting["configs"]["gamma_s"]:
                while len(stations_to_remove) < self.setting["configs"]["gamma_s"]:
                    index_removal = np.random.randint(0, len(possible_stations))
                    if possible_stations[index_removal] not in stations_to_remove:
                        stations_to_remove.append(possible_stations[index_removal])
            else:
                for station in possible_stations:
                    stations_to_remove.append(station)
        
        temporary_solution = solution.copy_solution(solution)
        remove_counter = 0
        for i, route in enumerate(temporary_solution.routes):
            for idx in route:
                if idx["name"] in stations_to_remove and remove_counter < self.setting["configs"]["gamma_s"]:
                    solution.remove_node(idx, i)
                    remove_counter += 1

        return stations_to_remove


class LongestWaitingTimeDestroyStation(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "LongestWaitingTimeDestroyStation"

    def apply(self, solution : Solution):
        stations_to_remove = []

        stations_to_remove = []
        possible_stations = []

        for route in solution.routes:
            for node in route:
                if node["isStation"]:
                    possible_stations.append(node["name"])

        if possible_stations != []:
            if len(possible_stations) > self.setting["configs"]["gamma_s"]:
                while len(stations_to_remove) < self.setting["configs"]["gamma_s"]:
                    expected_waiting_times = {}
                    for i, station in enumerate(self.setting["instance"].charging_stations):
                        if station["StringID"] in possible_stations:
                            expected_waiting_times.update({
                                                        station["StringID"] : self.setting["instance"].get_expected_waiting_time(i)
                                                        })
                    
                    sorted_station_times = dict(
                        sorted(
                            expected_waiting_times.items(), 
                            key = lambda item: item[1], 
                            reverse = True
                            )
                        )
                    
                    for station in sorted_station_times:
                        if station not in stations_to_remove and len(stations_to_remove) < self.setting["configs"]["gamma_s"]:
                            stations_to_remove.append(station)

            else:
                for station in possible_stations:
                    stations_to_remove.append(station)
        
        temporary_solution = solution.copy_solution(solution)
        remove_counter = 0
        for i, route in enumerate(temporary_solution.routes):
            for idx in route:
                if idx["name"] in stations_to_remove and remove_counter < self.setting["configs"]["gamma_s"]:
                    solution.remove_node(idx, i)
                    remove_counter += 1

        return stations_to_remove