# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt

class Plotter: # TODO: al posto di una classe metterei i metodi in instance, cosa ne dici?
    def __init__(self, instance, solution):
        self.instance = instance
        self.solution = solution

    def show_solution(self, filepath = None):
        g = nx.DiGraph()
        color_map = []

        for customer in self.instance.customers:
            g.add_node(customer["StringID"], pos = (customer["x"], customer["y"]))
            color_map.append('yellow')
        
        for station in self.instance.charging_stations:
            g.add_node(station["StringID"], pos = (station["x"], station["y"]))
            color_map.append('green')

        g.add_node(self.instance.start_point["StringID"], pos = (station["x"], station["y"]))
        color_map.append('red')

        edge_label = {}
        for j, route in enumerate(self.solution.routes):
            for i in range(len(route)-1):
                g.add_edge(route[i]["name"], route[i+1]["name"])
                edge_label.update({(route[i]["name"], route[i+1]["name"]) : str(j)})

        node_label = {}
        for j, route in enumerate(self.solution.routes):
            for i, element in enumerate(route):
                if i != 0:
                    node_label.update({element["name"] : str(int(self.solution.arrival_times[j][i-1]))})
        # TODO: cambiare colore per le varie routes

        pos = nx.get_node_attributes(g, 'pos')
        pos_nodes = self.nudge(pos, 0, -2.5)

        nx.draw(g, with_labels = True, pos = pos, node_color = color_map)
        nx.draw_networkx_edge_labels(g, pos = pos, edge_labels = edge_label)
        nx.draw_networkx_labels(g, pos = pos_nodes, labels = node_label)
        if filepath: # TODO: completare qui results
            pass# stampo su file
        else:
            plt.show()


    def nudge(self, pos, x_shift, y_shift):
        return {n:(x + x_shift, y + y_shift) for n,(x,y) in pos.items()}
        