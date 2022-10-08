# -*- coding: utf-8 -*-
import copy, math, time, random, joblib, statistics, pandas
import numpy as np
from sklearn.decomposition import dict_learning
from instance import *
import instance
from solver.ALNS_operators import *
from solver.plotter.plotter import Plotter
from solver.solution import Solution
import csv

moves_list = {  "customer_repair" : {"GreedyRepairCustomer" : 0},
                "customer_destroy" : {"GreedyDestroyCustomer" : 0,"WorstDistanceDestroyCustomer" : 1,"WorstTimeDestroyCustomer" : 2,"RandomRouteDestroyCustomer" : 3,
                                      "ZoneDestroyCustomer" : 4,"DemandBasedDestroyCustomer" : 5,"TimeBasedDestroyCustomer" : 6,"ProximityBasedDestroyCustomer" : 7,"ShawDestroyCustomer" : 8,
                                      "GreedyRouteRemoval" : 9,"ProbabilisticWorstRemovalCustomer" : 10},
                "station_repair" : {"DeterministicBestRepairStation" : 0,"ProbabilisticBestRepairStation" : 1},
                "station_destroy" : {"RandomDestroyStation" : 0,"LongestWaitingTimeDestroyStation" : 1}
                }

rf_models = {  "GreedyRepairCustomer" : joblib.load("../ML-results-MultiClassModels-3way/random_forest_GreedyRepairCustomer.joblib"),
                "GreedyDestroyCustomer" : joblib.load("../ML-results-MultiClassModels-3way/random_forest_GreedyDestroyCustomer.joblib"),
                "WorstDistanceDestroyCustomer" : joblib.load("../ML-results-MultiClassModels-3way/random_forest_WorstDistanceDestroyCustomer.joblib"),
                "WorstTimeDestroyCustomer" : joblib.load("../ML-results-MultiClassModels-3way/random_forest_WorstTimeDestroyCustomer.joblib"),
                "RandomRouteDestroyCustomer" : joblib.load("../ML-results-MultiClassModels-3way/random_forest_RandomRouteDestroyCustomer.joblib"),
                "ZoneDestroyCustomer" : joblib.load("../ML-results-MultiClassModels-3way/random_forest_ZoneDestroyCustomer.joblib"),
                "DemandBasedDestroyCustomer" : joblib.load("../ML-results-MultiClassModels-3way/random_forest_DemandBasedDestroyCustomer.joblib"),
                "TimeBasedDestroyCustomer" : joblib.load("../ML-results-MultiClassModels-3way/random_forest_TimeBasedDestroyCustomer.joblib"),
                "ProximityBasedDestroyCustomer" : joblib.load("../ML-results-MultiClassModels-3way/random_forest_ProximityBasedDestroyCustomer.joblib"),
                "ShawDestroyCustomer" : joblib.load("../ML-results-MultiClassModels-3way/random_forest_ShawDestroyCustomer.joblib"),
                "GreedyRouteRemoval" : joblib.load("../ML-results-MultiClassModels-3way/random_forest_GreedyRouteRemoval.joblib"),
                "ProbabilisticWorstRemovalCustomer" : joblib.load("../ML-results-MultiClassModels-3way/random_forest_ProbabilisticWorstRemovalCustomer.joblib"),
                "DeterministicBestRepairStation" : joblib.load("../ML-results-MultiClassModels-3way/random_forest_DeterministicBestRepairStation.joblib"),
                "ProbabilisticBestRepairStation" : joblib.load("../ML-results-MultiClassModels-3way/random_forest_ProbabilisticBestRepairStation.joblib"),
                "RandomDestroyStation" : joblib.load("../ML-results-MultiClassModels-3way/random_forest_RandomDestroyStation.joblib"),
                "LongestWaitingTimeDestroyStation" : joblib.load("../ML-results-MultiClassModels-3way/random_forest_LongestWaitingTimeDestroyStation.joblib")
                # 
                # 
                }

class Solver:
    def __init__(self, configs: dict, instance: Instance):
        self.instance = instance
        self.configs = configs
        self.operators_setup()

    def operators_setup(self):
        operators_configs = {
            "instance" : self.instance,
            "configs": self.configs, # TODO mettere a posto
            "updater" : self.update_solution_operators,
            "recourse_cost" : self.evaluate_second_stage,
            "customer_infeasible" : self.check_next_customer_infeasible,
            "route_feasible" : self.check_route_feasible_after_insertion,
            "check_feasibility" : self.check_solution_feasibility
        }
        customer_repair_operators = [
            GreedyRepairCustomer(operators_configs),
            # ProbabilisticGreedyRepairCustomer(operators_configs),
            # ProbabilisticGreedyConfidenceRepairCustomer(operators_configs)
        ]
        customer_destroy_operators = [
            GreedyDestroyCustomer(operators_configs),
            WorstDistanceDestroyCustomer(operators_configs),
            WorstTimeDestroyCustomer(operators_configs),
            RandomRouteDestroyCustomer(operators_configs),
            ZoneDestroyCustomer(operators_configs),
            DemandBasedDestroyCustomer(operators_configs),
            TimeBasedDestroyCustomer(operators_configs),
            ProximityBasedDestroyCustomer(operators_configs),
            ShawDestroyCustomer(operators_configs),
            GreedyRouteRemoval(operators_configs),
            ProbabilisticWorstRemovalCustomer(operators_configs)
        ]
        station_repair_operators = [
            DeterministicBestRepairStation(operators_configs),
            ProbabilisticBestRepairStation(operators_configs)
        ]
        station_destroy_operators = [
            RandomDestroyStation(operators_configs),
            LongestWaitingTimeDestroyStation(operators_configs)
        ]
        self.operators = {
            "customer_repair" : {
                "score" : [0] * len(customer_repair_operators) ,
                "weight" : [1/len(customer_repair_operators)] * len(customer_repair_operators),
                "operators" : customer_repair_operators
            },
            "customer_destroy" : {
                "score" : [0] * len(customer_destroy_operators),
                "weight" : [1/len(customer_destroy_operators)] * len(customer_destroy_operators),
                "operators" : customer_destroy_operators
            },
            "station_repair" : {
                "score" : [0] * len(station_repair_operators),
                "weight" : [1/len(station_repair_operators)] * len(station_repair_operators),
                "operators" : station_repair_operators
            },
            "station_destroy" : {
                "score" : [0] * len(station_destroy_operators),
                "weight" : [1/len(station_destroy_operators)] * len(station_destroy_operators),
                "operators" : station_destroy_operators
            }
        }

    def solve(self, ran_seed, collect_data=False, verbose=False):
        # TODO: ragionare su una versione vebose che dia dettagli sui calcoli che fa la soluzione
        # c = Solution(self.instance, self.configs)
        # c.add_route()
        max_retry = 100
        if self.configs["consider_pickup"] == "Yes":
            consider_pickup = True
        else:
            consider_pickup = False
        if self.configs["consider_partial"] == "Yes":
            consider_partial = True
        else:
            consider_partial = False
        
        # self.construct_initial_solution()
        it = 0
        T = self.configs["annealing_parameters"]["T_0"]
        feasible = False
        while not feasible and it < max_retry:
            x_0 = self.construct_initial_solution()
            if consider_partial:
                x_0 = self.partial_recharge_update(x_0)   
            if self.check_solution_feasibility(x_0):
                feasible = True

        
        plotter = Plotter(self.instance, x_0)
        # plotter.show_solution()

        x_best = x_0.copy_solution(x_0)
        x_current = x_0.copy_solution(x_0)
        x_previous = x_0.copy_solution(x_0) 
        OF_x_0 = self.instance.compute_OF(x_0)
        OF_x_best = OF_x_0
        OF_x_current = OF_x_0
        OF_x_previous = OF_x_0

        self.input_vector = [0.0 for i in range(18)]
        ofis = 0

        # initialize score and probabilities of operators as 1/num_operators
        print("Starting solution OF: ", OF_x_0)
        
        # generate random utilization level for each station
        # compute expected_waiting_time at every station

        diffiterationvectnew=[]

        startTime = time.time()
        for iteration in range(1, self.configs["number_iterations"] + 1):
            print("Iteration no", iteration, "of", self.configs["number_iterations"])
            
            algorithms_applied = self.apply_destroy_and_repair(
                x_current, iteration
            )

            OF_x_current = self.instance.compute_OF(x_current)
            # apply second stage and compute recourse cost
            
            # TODO: in algoritmi complicati aggiungi commenti come quelli sotto per aumentare la chiarezza
            # SOLUTION UPDATE
            if self.check_solution_feasibility(x_current):
                recourse_cost = self.apply_second_stage_evaluation(x_current, consider_pickup, consider_partial)
                OF_x_current += recourse_cost
                if OF_x_current < OF_x_previous:
                    x_previous = copy.deepcopy(x_current)
                    OF_x_previous = OF_x_current
                    self.update_score_operator(self.operators, self.configs["alns_scores"]["current_better"], algorithms_applied)
                    if OF_x_previous < OF_x_best:
                        x_best = copy.deepcopy(x_current)
                        OF_x_best = OF_x_current
                        self.update_score_operator(self.operators, self.configs["alns_scores"]["global_best"], algorithms_applied)
                else:
                    OF_difference = OF_x_current - OF_x_previous
                    r = np.random.uniform()
                    if r < math.exp((-OF_difference) / (self.configs["annealing_parameters"]["k"] * T)):
                        x_previous = copy.deepcopy(x_current)
                        OF_x_previous = OF_x_current
                        self.update_score_operator(self.operators, self.configs["alns_scores"]["solution_accepted"], algorithms_applied)
                        T = T / ( 1 + ( T * self.configs["annealing_parameters"]["frazionamento"] ) )
                    else:
                        self.update_score_operator(self.operators, self.configs["alns_scores"]["solution_rejected"], algorithms_applied)
                        x_current = copy.deepcopy(x_previous)
            else:
                self.update_score_operator(self.operators, self.configs["alns_scores"]["solution_rejected"], algorithms_applied)
                x_current = copy.deepcopy(x_previous)

            # UPDATE WEIGHT OPERATOR CUSTOMER
            if (iteration % self.configs["number_iterations_customer_op_update_weights"]) == 0:
                for i in range(len(self.operators["customer_repair"]["operators"])):
                    self.update_weights(
                        self.operators["customer_repair"]["weight"],
                        i,
                        self.operators["customer_repair"]["score"]
                    )
                
                for i in range(len(self.operators["customer_destroy"]["operators"])):
                    self.update_weights(
                        self.operators["customer_destroy"]["weight"],
                        i,
                        self.operators["customer_destroy"]["score"]
                    )
                self.wipe_scores("customer")

            # UPDATE WEIGHT OPERATOR STATION
            if (iteration % self.configs["number_iterations_station_op_update_weights"]) == 0:

                for i in range(len(self.operators["station_repair"]["operators"])):
                    self.update_weights(
                        self.operators["station_repair"]["weight"],
                        i,
                        self.operators["station_repair"]["score"]
                    )
                for i in range(len(self.operators["station_destroy"]["operators"])):
                    self.update_weights(
                        self.operators["station_destroy"]["weight"],
                        i,
                        self.operators["station_destroy"]["score"]
                    ) 
                
                self.wipe_scores("station")

            if (iteration % self.configs["number_iterations_waiting_update"]) == 0:
                # TODO: completare?
                # compute waiting time parameter alfa, update waiting time
                # if alfa > 1:
                #   apply solution correction (?)
                pass
            print(f"\t {OF_x_current:.2f} {OF_x_best:.2f}")
            if collect_data:
                pass # CODICE RAGAZZI

            finishTime = time.time()
        
            # OFIS
            if(iteration == 1):
                self.input_vector[0] = OF_x_0
            else:
                self.input_vector[0] = ofis
            
            # OFFS
            self.input_vector[1] = self.instance.compute_OF(x_current)
            ofis = self.instance.compute_OF(x_current)

            diffiterationvectnew.append(self.input_vector[0]-self.input_vector[1])

            # Exe_Time_d-r
            self.input_vector[2] = finishTime-startTime

            # Avg_Battery_Status
            global_single_nodes_sum = 0
            for route in x_current.routes:
                single_nodes = []
                for node in route:
                    single_nodes.append(node)
                
                for i in range(len(single_nodes)-1):
                    global_single_nodes_sum += self.instance.distance_matrix[single_nodes[i]["name"]][single_nodes[i+1]["name"]]
            
            self.input_vector[3] = global_single_nodes_sum / len(x_current.routes)
            
            # Avg_SoC
            soc_sum = 0
            for single_soc in x_current.vehicles:
                soc_sum += single_soc["SoC"]
            
            self.input_vector[4] = soc_sum / len(x_current.vehicles)

            #print(x_current.routes)
            
            # Avg_Num_Charge
            count = 0
            for route in x_current.routes:
                for node in route:
                    if node["name"][0] == 'S':
                        count += 1

            self.input_vector[5] = count / len(x_current.vehicles)
            
            # Avg_Vehicle_Capacity
            Current_cargo_sum_c = 0
            for c in x_current.vehicles:
                Current_cargo_sum_c += c["current_cargo"]
            
            self.input_vector[6] = Current_cargo_sum_c  / len(x_current.vehicles)

            # Avg_Customer_Demand
            sum_tot_demand = 0
            for customer in self.instance.customers:
                sum_tot_demand += customer["demand"]
            
            self.input_vector[7] = sum_tot_demand / len(self.instance.customers)
            
            # Num_Vehicles
            self.input_vector[8] = len(x_current.vehicles)

            # Avg_Service_Time
            sum_tot_service_time = 0
            for customer in self.instance.customers:
                sum_tot_service_time += customer["ServiceTime"]
            
            self.input_vector[9] = sum_tot_service_time / len(self.instance.customers)

            # Avg_Customer_TimeWindow & Var_Customer_TimeWindow
            sum_tot_customer_timeWindow = []
            for customer in self.instance.customers:
                sum_tot_customer_timeWindow.append(customer["DueDate"] - customer["ReadyTime"])
            
            self.input_vector[10] = statistics.mean(sum_tot_customer_timeWindow)
            self.input_vector[11] = statistics.variance(sum_tot_customer_timeWindow)
            
            # Avg_Customer_customer_min_dist & Var_Customer_customer_min_dist
            vectordist = []
            for c1 in self.instance.customers:
                temp = 1000000
                for c2 in self.instance.customers:
                    if c1 != c2:
                        dist=math.sqrt(pow((c1['x']-c2['x']),2)+pow((c1['y']-c2['y']),2))
                        if  dist<temp:
                            temp=dist
                vectordist.append(temp)
            self.input_vector[12] = statistics.mean(vectordist)
            self.input_vector[13] = statistics.variance(vectordist)

            # Avg_Customer_station_min_dist & Var_Customer_station_min_dist
            vectordiststation = []
            for c1 in self.instance.customers:
                temp = 1000000
                for s1 in self.instance.charging_stations:
                    if c1 != s1:
                        dist=math.sqrt(pow((c1['x']-s1['x']),2)+pow((c1['y']-s1['y']),2))
                        if  dist<temp:
                            temp=dist
                vectordiststation.append(temp)
            self.input_vector[14] = statistics.mean(vectordiststation)
            self.input_vector[15] = statistics.variance(vectordiststation)

            # Avg_Customer_deposit_dist & Var_Customer_deposit_dist
            vectordistdeposit = []
            for c1 in self.instance.customers:
                dist=(math.sqrt(pow((c1['x']-self.instance.start_point['x']),2)+pow((c1['y']-self.instance.start_point['y']),2)))
                vectordistdeposit.append(dist)
            self.input_vector[16] = statistics.mean(vectordistdeposit)
            self.input_vector[17] = statistics.variance(vectordistdeposit)

        print("Starting solution OF: ", OF_x_0)
        print("Final solution OF: ", OF_x_best)
        print("Total elapsed time: ", time.time() - startTime)
        # plotter = Plotter(self.instance, x_best)
        # plotter.show_solution()

        import json
        settings = json.load(open("./etc/settings.json"))
        filename = settings["instance_file_name"]
        modelresultsW = open('./model_results_3way.csv', 'a', newline='')
        writerMR = csv.writer(modelresultsW)
        writerMR.writerow([filename,ran_seed,OF_x_0,OF_x_best,OF_x_0-OF_x_best,time.time() - startTime,diffiterationvectnew])
        modelresultsW.close()

        return x_best, OF_x_best

    def update_score_operator(self, operators, which_score, algorithms_applied):
        if algorithms_applied["station"]["repair"]["number"] != 999:
            operators["station_destroy"]["score"][algorithms_applied["station"]["destroy"]["number"]] += which_score
            operators["station_repair"]["score"][algorithms_applied["station"]["repair"]["number"]] += which_score
        else:
            operators["customer_destroy"]["score"][algorithms_applied["customer"]["destroy"]["number"]] += which_score
            operators["customer_repair"]["score"][algorithms_applied["customer"]["repair"]["number"]] += which_score

    def update_weights(self, operator, apply_on, score):
        operator[apply_on] = (operator[apply_on] * self.configs["alns_decay_parameter"]) + ( 1 - self.configs["alns_decay_parameter"]) * score[apply_on]

    def wipe_scores(self, which):
        for i in range(len(self.operators[which + "_destroy"]["operators"])):
            self.operators[which + "_destroy"]["score"][i] = 0
        for i in range(len(self.operators[which + "_repair"]["operators"])):
            self.operators[which + "_repair"]["score"][i] = 0

    def _select_method(self, op_list, weights):
        pos = random.choices(
            range(len(op_list)),
            weights
        )[0]
        return op_list[pos], pos 

    def apply_destroy_and_repair(self, x: Solution, iteration: int):
        couples = {
            "station" : {
                "destroy" : {
                    "number" : 999,
                    "time" : 0
                },
                "repair" : {
                    "number" : 999,
                    "time" : 0
                }
            },
            "customer" : {
                "destroy" : {
                    "number" : 0,
                    "time" : 0 
                },
                "repair" : {
                    "number" : 0,
                    "time" : 0
                },
            }
        }
        if (iteration % self.configs["number_iterations_station_removal"]) == 0:
            # VECCHIO CODICE ALEX - START
            """ # SELECT DESTROY STATION
            station_destroy, pos_destroy = self._select_method(
                self.operators["station_destroy"]["operators"],
                self.operators["station_destroy"]["weight"]
            )

            # SELECT REPAIR STATION
            station_repair, pos_repair = self._select_method(
                self.operators["station_repair"]["operators"],
                self.operators["station_destroy"]["weight"]
            ) """
            # VECCHIO CODICE ALEX - FINISH 

            input_vector_2D = [self.input_vector]

            # SELECT DESTROY STATION
            _, pos_destroy = self._select_method(
                self.operators["station_destroy"]["operators"],
                self.operators["station_destroy"]["weight"]
            )
            
            predicts = {}
            for move in moves_list["station_destroy"].keys():
                loaded_rf = rf_models[move]
                predicts[''+move] = loaded_rf.predict(input_vector_2D)
            #print(predicts)
            
            if "Very Good" in predicts.values():
                station_destroy = random.choice([i for i in predicts if predicts[i][0]=="Very Good"])
            elif "Good" in predicts.values():
                station_destroy = random.choice([i for i in predicts if predicts[i][0]=="Good"])
            else:
                station_destroy = random.choice([i for i in predicts if predicts[i][0]=="Bad"])
            station_destroy = self.operators["station_destroy"]["operators"][moves_list["station_destroy"][station_destroy]]

            # SELECT REPAIR STATION
            _, pos_repair = self._select_method(
                self.operators["station_repair"]["operators"],
                self.operators["station_destroy"]["weight"]
            )

            predicts = {}
            for move in moves_list["station_repair"].keys():
                loaded_rf = rf_models[move]
                predicts[''+move] = loaded_rf.predict(input_vector_2D)
            #print(predicts)
            
            if "Very Good" in predicts.values():
                station_repair = random.choice([i for i in predicts if predicts[i][0]=="Very Good"])
            elif "Good" in predicts.values():
                station_repair = random.choice([i for i in predicts if predicts[i][0]=="Good"])
            else:
                station_repair = random.choice([i for i in predicts if predicts[i][0]=="Bad"])
            station_repair = self.operators["station_repair"]["operators"][moves_list["station_repair"][station_repair]]

            # APPLY DESTROY STATION
            currtime = time.time()
            removed_stations = station_destroy.apply(x)
            time_elapsed_destroy = time.time() - currtime
            couples["station"]["destroy"]["number"] = pos_destroy
            couples["station"]["destroy"]["time"] = time_elapsed_destroy
            
            # APPLY REPAIR STATION
            currtime = time.time()
            station_repair.apply(x, removed_stations)
            time_elapsed_repair = time.time() - currtime
            couples["station"]["repair"]["number"] = pos_repair
            couples["station"]["repair"]["time"] = time_elapsed_repair

            print(f'>> Applying STATION destroy [{station_destroy} {time_elapsed_destroy:.2f}] and repair [{station_repair} {time_elapsed_repair:.2f}]')

        # VECCHIO CODICE ALEX - START
        """ # SELECT DESTROY CUSTOMER
        customer_destroy, pos_destroy = self._select_method(
            self.operators["customer_destroy"]["operators"], 
            self.operators["customer_destroy"]["weight"]
        )
        # SELECT REPAIR CUSTOMER
        customer_repair, pos_repair = self._select_method(
            self.operators["customer_repair"]["operators"],
            self.operators["customer_repair"]["weight"]
        ) """
        # VECCHIO CODICE ALEX - FINISHED

        if iteration == 1:
            # SELECT DESTROY CUSTOMER
            customer_destroy, pos_destroy = self._select_method(
                self.operators["customer_destroy"]["operators"], 
                self.operators["customer_destroy"]["weight"]
            )
            # SELECT REPAIR CUSTOMER
            customer_repair, pos_repair = self._select_method(
                self.operators["customer_repair"]["operators"],
                self.operators["customer_repair"]["weight"]
            )
        else:
            input_vector_2D = [self.input_vector]

            # SELECT DESTROY CUSTOMER
            _, pos_destroy = self._select_method(
                self.operators["customer_destroy"]["operators"], 
                self.operators["customer_destroy"]["weight"]
            )

            predicts = {}
            for move in moves_list["customer_destroy"].keys():
                loaded_rf = rf_models[move]
                predicts[''+move] = loaded_rf.predict(input_vector_2D)
            #print(predicts)
                
            if "Very Good" in predicts.values():
                customer_destroy = random.choice([i for i in predicts if predicts[i][0]=="Very Good"])
            elif "Good" in predicts.values():
                customer_destroy = random.choice([i for i in predicts if predicts[i][0]=="Good"])
            else:
                customer_destroy = random.choice([i for i in predicts if predicts[i][0]=="Bad"])
            customer_destroy = self.operators["customer_destroy"]["operators"][moves_list["customer_destroy"][customer_destroy]]

            # SELECT REPAIR CUSTOMER
            customer_repair, pos_repair = self._select_method(
                self.operators["customer_repair"]["operators"],
                self.operators["customer_repair"]["weight"]
            )

            """ predicts = {}
            for move in moves_list["customer_repair"].keys():
                loaded_rf = rf_models[move]
                predicts[''+move] = loaded_rf.predict(input_vector_2D)
            #print(predicts)
                
            if "Very Good" in predicts.values():
                customer_repair = random.choice([i for i in predicts if predicts[i][0]=="Very Good"])
            elif "Good" in predicts.values():
                customer_repair = random.choice([i for i in predicts if predicts[i][0]=="Good"])
            else:
                customer_repair = random.choice([i for i in predicts if predicts[i][0]=="Bad"])
            customer_repair = self.operators["customer_repair"]["operators"][moves_list["customer_repair"][customer_repair]] """


        # APPLY REMOVE CUSTOMER
        currtime = time.time()
        removed_customers = customer_destroy.apply(x)
        time_elapsed_destroy = time.time() - currtime
        couples["customer"]["destroy"]["number"] = pos_destroy
        couples["customer"]["destroy"]["time"] = time_elapsed_destroy
        
        # APPLY REPAIR CUSTOMER
        currtime = time.time()
        customer_repair.apply(x, removed_customers)
        time_elapsed_repair = time.time() - currtime
        couples["customer"]["repair"]["number"] = pos_repair
        couples["customer"]["repair"]["time"] = time_elapsed_repair

        print(f'Applying CUSTOMER destroy [{customer_destroy} {time_elapsed_destroy:.2f}] and repair [{customer_repair} {time_elapsed_repair:.2f}]')
        x.remove_empty_routes()

        return couples

    def apply_second_stage_evaluation(self, solution: Solution, consider_pickup = False, consider_partial = False):
        E_cost_k = [0] * len(solution.routes)
        for i in range(len(solution.routes)):
            E_cost_k[i] = self.evaluate_second_stage(solution, i, consider_pickup, consider_partial)
        
        return (np.mean(np.array(E_cost_k)))

    def evaluate_second_stage(self, first_stage_solution: Solution,  route_idx: int, consider_pickup=False, consider_partial = False):
        first_stage_solution.generate_station_info_route(route_idx)
        old_waiting_time = self.instance.E_recharge
        E_cost_k = 0
        route = first_stage_solution.routes[route_idx]
        tmp_solution = first_stage_solution.copy_solution(first_stage_solution)
        cost_k = 0
        station_index = False

        tmp_solution.station_info[route_idx]
        for station_index in tmp_solution.station_info[route_idx]:
            # station_index = tmp_solution.station_info[route_idx]

            route_energy = self.instance.compute_energy_cost(route)
            if station_index < len(route):
                if station_index != False:
                    station_arrival = tmp_solution.arrival_times[route_idx][station_index - 1]
                    for _ in range(self.configs["n_scenarios"]):
                        # CALCOLO NUOVI TEMPI DI ATTESA
                        tmp_solution.update_station_waiting_time(
                            route_idx, old_waiting_time
                        )
                        # estraggo i customer dopo la stazione e i loro tempi, escludendo l'ultimo elemento che è D0
                        customers_after_station = route[station_index+1:]
                        customers_after_station = customers_after_station[:-1]
                        customer_times_after_station = tmp_solution.arrival_times[route_idx][station_index:]
                        customer_times_after_station = customer_times_after_station[:-1]

                        # prendo le finestre late dei customer
                        customer_due_date = []
                        for el, customer in enumerate(customers_after_station):
                            if customer["isCustomer"]:
                                customer_due_date.append(self.instance.customers_dict[customer["name"]]["DueDate"])
                            elif customer["isStation"]:
                                customer_times_after_station.pop(el)

                        customer_due_date = np.array(customer_due_date)
                        customer_times_after_station = np.array(customer_times_after_station)
                        # trovo quali finestre sono state violate (quelli != 0)
                        violated = np.maximum(customer_times_after_station - customer_due_date, np.zeros(len(customer_due_date)))

                        for j, element in enumerate(violated):
                            if element != 0 and customers_after_station[j]["isCustomer"]:
                                
                                if consider_partial: # try partial recharge recourse
                                    cost_recourse = self.partial_recharge_recourse(
                                        tmp_solution,
                                        customers_after_station[j],
                                        station_arrival,
                                        route_idx,
                                        route_energy
                                    )
                                    if cost_recourse[1]: # if successful, save cost
                                        cost_k += cost_recourse[0]
                                    else: # if unsuccessful  try pickup exchange
                                        if consider_pickup:
                                            cost_recourse = self.pickup_exchange_recourse(
                                                tmp_solution,
                                                customers_after_station[j],
                                                station_arrival,
                                                route_idx,
                                                route_energy
                                            )
                                            if cost_recourse[1]: # if successful save cost
                                                cost_k += cost_recourse[0]
                                            else: # if unsuccessful use default recourse
                                                cost_k += self.new_route_recourse(
                                                    tmp_solution,
                                                    customers_after_station[j],
                                                    station_arrival,
                                                    route_idx,
                                                    route_energy
                                                )
                                        else: # if not consider pickup, use default recourse
                                            cost_k += self.new_route_recourse(
                                                                tmp_solution,
                                                                customers_after_station[j],
                                                                station_arrival,
                                                                route_idx,
                                                                route_energy
                                                            )
                                                    
                                elif consider_pickup: # try pickup exchange
                                    cost_recourse = self.pickup_exchange_recourse(
                                        tmp_solution,
                                        customers_after_station[j],
                                        station_arrival,
                                        route_idx,
                                        route_energy
                                    )
                                    if cost_recourse[1]: # if successful save cost
                                        cost_k += cost_recourse[0]
                                    else: # if unsuccessful use default recourse
                                        cost_k += self.new_route_recourse(
                                            tmp_solution,
                                            customers_after_station[j],
                                            station_arrival,
                                            route_idx,
                                            route_energy
                                        )
                                        
                                
                                else: # if no exchange nor partial recharge, use default recourse
                                # se l'elemento violato è un customer, faccio partire una nuova route dal tempo dell'arrivo alla stazione
                                    cost_k += self.new_route_recourse(
                                            tmp_solution,
                                            customers_after_station[j],
                                            station_arrival,
                                            route_idx,
                                            route_energy
                                        )
                                    

                            # se il veicolo va in overtime, calcolo il prezzo da pagare overtime
                            if tmp_solution.vehicles[route_idx]["time"] > self.instance.start_point["DueDate"]:
                                cost_k = cost_k +\
                                    self.instance.cost_driver*(self.instance.start_point["DueDate"] - tmp_solution.vehicles[route_idx]["time"])+\
                                    self.instance.cost_overtime*(tmp_solution.vehicles[route_idx]["time"] - self.instance.start_point["DueDate"])
                            else:
                                cost_k = cost_k +\
                                    self.instance.cost_driver * (tmp_solution.vehicles[route_idx]["time"] - self.instance.start_point["DueDate"])
                # aggiungo il valore atteso di costo di ogni route
        E_cost_k = cost_k / self.configs["n_scenarios"]

        return E_cost_k

    def pickup_exchange_recourse(self, solution : Solution, violated : dict, station_arrival : float, route_idx, route_energy) -> float:
        # get time and idx of the elements at the moment the recourse vehicle arrives at the station
        route_times = {}
        for i, route in enumerate(solution.arrival_times):
            if i != route_idx:
                for j, time in enumerate(route):
                    if time > station_arrival:
                        route_times.update({
                            (i, j + 1) : time - station_arrival
                        })
                        break
        # sort it in increasing order (lowest time difference first)
        sorted_times = dict(sorted(route_times.items(), key = lambda item: item[1]))
        route_position = list(sorted_times.keys())
        # try to add the violated customer to each route and see if route is still feasible
        for couple in route_position:
            route = couple[0]
            position = couple[1]
            solution.add_node(
                node = violated["name"],
                route_idx = route,
                position_idx = position + 1
            )
            # if the route is still feasible, compute recourse cost and go to next route
            if self.check_solution_feasibility(solution):
                energy_difference = route_energy - self.instance.compute_energy_cost(
                    solution.routes[route_idx]
                )
                cost_k = self.instance.compute_cost_route(
                    solution.routes[route],
                    solution.arrival_times[route]
                ) - (self.instance.cost_energy * energy_difference)
                return (cost_k, True)
        return (0, False)

    def new_route_recourse(self, tmp_solution : Solution, violated : dict, station_arrival : float, route_idx, route_energy) -> float:
        tmp_solution.remove_node(violated, route_idx)
        # creo route e aggiungo nodo e depot
        tmp_solution.add_route_time(station_arrival)
        tmp_solution.add_node(
            node = violated["name"], 
            route_idx = len(tmp_solution.routes) - 1,
            position_idx = 1
        )
        tmp_solution.add_node(
            node = self.instance.start_point["StringID"],
            route_idx = len(tmp_solution.routes) - 1,
            position_idx = 2
        )
        # calcolo l'energia saved con questa operazione
        energy_difference = route_energy - self.instance.compute_energy_cost(
            tmp_solution.routes[route_idx]
        )
        cost_k = self.instance.compute_cost_route(
            tmp_solution.routes[-1],
            tmp_solution.arrival_times[-1]
        ) - (self.instance.cost_energy * energy_difference)
        return cost_k

    def partial_recharge_recourse(self, tmp_solution : Solution, violated : dict, station_arrival : float, route_idx, route_energy):
        tmp_solution = self.partial_recharge_update_single_route(tmp_solution, 
                                                                route_idx, 
                                                                self.configs["SoC_tolerance"])
        if self.check_solution_feasibility(tmp_solution):
            energy_difference = route_energy - self.instance.compute_energy_cost(
                tmp_solution.routes[route_idx]
            )
            cost_k = self.instance.compute_cost_route(
                tmp_solution.routes[route_idx],
                tmp_solution.arrival_times[route_idx]
            ) - (self.instance.cost_energy * energy_difference)
            return (cost_k, True)
        return (0, False)


    def update_solution_operators(self, solution, which_route, customer_idx):
        # TODO:[future] questo metodo dovrebbe essere messo nella classe Solution e la classe integrata meglio. Però facciamolo quando abbiamo una prima versione funzionante di tutto, ok?
        routes = solution["routes"]
        vehicles_list = solution["vehicles"]
        vehicles_list[which_route] = dict(self.instance.base_electric_vehicle)
        # aggiornare in vehicles da customer_idx-1 to end

        for i in range(len(routes[which_route])):
            if i != 0:
                if "C" in routes[which_route][i]:
                    self.update_ev_customer(which_route, routes, vehicles_list, i)
                elif "S" in routes[which_route][i]:
                    self.update_ev_station(which_route, routes, vehicles_list, i)
                elif "D" in routes[which_route][i]:
                    self.update_ev_depot(which_route, routes, vehicles_list, i)

    def check_solution_feasibility(self, solution):
        # TODO: [future] questo metodo dovrebbe essere messo nella classe Solution, ma come sopra...
        for vehicle in solution.vehicles:
            if vehicle["SoC"] < 0.1:
                return False
            elif vehicle["current_cargo"] > self.instance.C:
                return False
            # elif vehicle["time"] > vehicle["depot_closure"]:
            #     return False
        return True

    def check_next_customer_infeasible(self, solution: Solution, customer_pos: list):
        prob = 0
        n_scenarios = self.configs["n_scenarios"]
        E_service_time = self.instance.E_recharge

        for _ in range(1, n_scenarios):
            base_solution = solution.copy_solution(solution)
            # put here generation of service time and update arrivals
            base_solution.update_station_waiting_time(customer_pos[0], E_service_time)
            successor_found = False

            if base_solution.routes[customer_pos[0]][customer_pos[1]+1]["isCustomer"]:
                successor_found = True
                time_successor = self.instance.customers_dict[base_solution.routes[customer_pos[0]][customer_pos[1]+1]["name"]]["DueDate"]

            if not successor_found:
                time_successor = self.instance.start_point["DueDate"]

            if base_solution.arrival_times[customer_pos[0]][customer_pos[1]] > time_successor:
                prob += 1
             
        prob = prob / n_scenarios
        return prob

    def check_route_feasible_after_insertion(self, solution : Solution, route_idx):
        it = 1
        prob = 0
        n_scenarios = self.configs["n_scenarios"]
        E_service_time = self.instance.E_recharge

        while it <= n_scenarios:
            base_solution = solution.copy_solution(solution)
            base_solution.update_station_waiting_time(route_idx, self.instance.E_recharge)
            base_solution.generate_station_info_route(route_idx)

            for i, node in enumerate(base_solution.routes[route_idx]):
                for station_index in base_solution.station_info[route_idx]:
                    if i > station_index and not node["isDepot"]:
                        customer_time = self.instance.customers_dict[node["name"]]["DueDate"]

                        if base_solution.arrival_times[route_idx][i-1] >=  customer_time:
                            prob += 1
                            break

            it += 1

        prob = 1 - prob / n_scenarios

        return prob

    def partial_recharge_update(self, x: Solution):
        for i in range(len(x.routes)):
            x = self.partial_recharge_update_single_route(x, i, self.configs["SoC_tolerance"])

        return x

    def partial_recharge_update_single_route(self, x : Solution, route_idx : int, SoC_tolerance = 0.15) -> Solution:
        for station_idx in x.station_info[route_idx]:
            total_distance = 0
            for j in range(len(x.routes[route_idx]) - 1, station_idx, -1):
                total_distance += self.instance.distance_matrix[x.routes[route_idx][j]["name"]][x.routes[route_idx][j-1]["name"]]
            SoC_consumed = (self.instance.h * total_distance) / self.instance.Q 
            target_SoC = SoC_consumed + SoC_tolerance
            if target_SoC > 0.9:
                target_SoC = 0.9
            x.update_route(route_idx, partial = True, target_SoC = target_SoC)
        
        return x


    def construct_initial_solution(self) -> Solution:

        """Solves the first stage solution.
        Steps are as follows:
        1. Get all customers in a stack, take one at a time and add them to a route. If an EV is near its due depot time
          it looks for a station if it needs charge, otherwise gets back to a depot and the next EV starts its route.
        2. If en route the EV needs charge, instead of going to a customer it looks for an available recharge station,
          recharging 40% of the charge (as mentioned in Keskin) then restarts. The recharging stations have a random
          waiting time, that gets realized when the ev reaches the station.
        3. Then the ev waits how much it has to wait and restarts its route after waiting and recharging.


        Returns:
            (routes, vehicles): the solution to the first step is returned as a dict, in [0] the list of routes used and in [1]
                the list of dictionaries of vehicles that have served, with a field that represents the list of times at which it reached a point
                in the route
        """

        run = True
        customer_name_lst = list(self.instance.customers_dict.keys())
        
        solution = Solution(self.instance)
        route_nmb = 0
        node_index = 1
        solution.add_fresh_route()
        
        while run:
            if solution.routes[route_nmb][node_index - 1]["isDepot"]:
                customer_to_add = self.get_nearest_customer(solution.routes[route_nmb][-1]["name"], solution.vehicles[route_nmb], customer_name_lst)
                solution.add_node(customer_to_add, route_nmb, node_index)
                customer_name_lst.remove(customer_to_add)
                node_index += 1
            else:
                customer_to_add = self.get_nearest_customer(solution.routes[route_nmb][-1]["name"], solution.vehicles[route_nmb], customer_name_lst)
                if customer_to_add != "D0":
                    potential_distance_travelled = self.instance.distance_matrix[solution.routes[route_nmb][-1]["name"]][customer_to_add]
                    new_soc = (solution.vehicles[route_nmb]["SoC"]*self.instance.Q - self.instance.h * potential_distance_travelled) / self.instance.Q
                    if new_soc < 0.1:
                        station_to_add = self.get_nearest_station(solution.routes[route_nmb][-1]["name"])
                        solution.add_node(station_to_add, route_nmb, node_index)
                        node_index += 1
                    else:
                        customer_to_add = self.get_nearest_customer(solution.routes[route_nmb][-1]["name"], solution.vehicles[route_nmb], customer_name_lst)
                        solution.add_node(customer_to_add, route_nmb, node_index)
                        customer_name_lst.remove(customer_to_add)
                        node_index += 1
                else:
                    customer_to_add = self.instance.start_point["StringID"]
                    potential_distance_travelled = self.instance.distance_matrix[solution.routes[route_nmb][-1]["name"]][customer_to_add]
                    new_soc = (solution.vehicles[route_nmb]["SoC"]*self.instance.Q - self.instance.h * potential_distance_travelled) / self.instance.Q
                    if new_soc < 0.1:
                        station_to_add = self.get_nearest_station(solution.routes[route_nmb][-1]["name"])
                        solution.add_node(station_to_add, route_nmb, node_index)
                        node_index += 1
                    else:
                        solution.add_node(customer_to_add, route_nmb, node_index)
                        if customer_name_lst == []:
                            run = False
                        else:
                            route_nmb += 1
                            solution.add_fresh_route()
                            node_index = 1
                            

        return solution

    def get_nearest_customer(self, start_idx, electric_vehicle, customer_name_lst):
        # find the next customer with the greedy insertion
        infeasible = True
        infeasible_list = []
        # now find the index of the min, excluding the values with 0, needs to be a C
        # check for availability on adding, if the next customer cannot be served because the EV would arrive late,
        # then the customer is ignored. If the next customer would bring the EV over cargo limit, it is ignored.
        # If the stopping conditions are met, the function returns the string "end_time" and the EV prepares for return to depot
        while infeasible: # TODO: anche qui, mi piacerebbe di più fare un for sui customer
            start_point_series = self.instance.distance_matrix[start_idx]
            # now find the index of the min, excluding the values with 0, needs to be a C
            # remove useless columns from series
            remove_cols = []
            for idx in start_point_series.axes[0]:
                is_depot = "D" in idx
                is_station = "S" in idx
                if is_depot or is_station or idx in infeasible_list or idx not in customer_name_lst:
                    
                    if idx not in remove_cols:
                        remove_cols.append(idx)
                    
            if start_point_series.drop(remove_cols).empty:
                return "D0"
            
            # get nearest customer to starting point for the greedy insertion after removing infeasible customers and depot/station
            potential_customer = start_point_series.drop(remove_cols).idxmin() 
            distance_travelled = self.instance.distance_matrix[start_idx][potential_customer]
            
            # check for customer feasibility in terms of time, cargo
            for customer in self.instance.customers:
                if potential_customer in customer["StringID"]:
                    due_date = customer["DueDate"]
                    cargo_weigth = customer["demand"]
            if (distance_travelled/self.instance.average_velocity + electric_vehicle["time"]) > due_date or (electric_vehicle["current_cargo"]+cargo_weigth)>self.instance.C:
                infeasible_list.append(potential_customer)
            else:
                infeasible = False

        return potential_customer

    def get_nearest_station(self, start_idx):
        start_point_series = self.instance.distance_matrix[start_idx] # now find the index of the min, excluding the values with 0, needs to be a C
        
        # remove useless columns from series
        remove_cols = []
        for idx in start_point_series.axes[0]:
            if "D" in idx or "C" in idx:
                try:
                    remove_cols.append(idx)
                except:
                    pass

        if start_point_series.drop(remove_cols).empty:
            return "end_time" 
        
        potential_station = start_point_series.drop(remove_cols).idxmin() 
        return potential_station
