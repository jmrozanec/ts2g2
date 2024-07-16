import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

import os
import csv
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from timeseries.strategies import TimeseriesToGraphStrategy, TimeseriesEdgeVisibilityConstraintsNatural, TimeseriesEdgeVisibilityConstraintsHorizontal, EdgeWeightingStrategyNull
from generation.strategies import RandomWalkWithRestartSequenceGenerationStrategy, RandomWalkSequenceGenerationStrategy, RandomNodeSequenceGenerationStrategy, RandomNodeNeighbourSequenceGenerationStrategy, RandomDegreeNodeSequenceGenerationStrategy
from core import model 
from sklearn.model_selection import train_test_split
import itertools

import xml.etree.ElementTree as ET
import random

#data
apple_data = pd.read_csv(os.path.join(os.getcwd(), "apple", "APPLE.csv"))
apple_data["Date"] = pd.to_datetime(apple_data["Date"])
apple_data.set_index("Date", inplace=True)

amazon_data = pd.read_csv(os.path.join(os.getcwd(), "amazon", "AMZN.csv"))
amazon_data["Date"] = pd.to_datetime(amazon_data["Date"])
amazon_data.set_index("Date", inplace=True)

segment_apple = apple_data[60:120]
segment_amazon = amazon_data[5629:5689]

#drawing time series
def plot_timeseries(sequence, title, x_legend, y_legend, color):
    plt.figure(figsize=(10, 6))
    plt.plot(sequence, linestyle='-', color=color)
    
    plt.title(title)
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)
    plt.grid(True)
    #plt.show()

def plot_timeseries_sequence(df_column, title, x_legend, y_legend, color):
    sequence = model.Timeseries(model.TimeseriesArrayStream(df_column)).to_sequence()
    plot_timeseries(sequence, title, x_legend, y_legend, color)


#drawing two time series simultaneously
def plot_two_timeseries(sequence1, sequence2, title, x_legend, y_legend, color1, color2):

    plot_apple = plt.subplot2grid((2,2), (0,0))
    plot_amazon = plt.subplot2grid((2,2), (0,1))
    plot_both = plt.subplot2grid((2,2), (1,0), colspan=2)


    plt.figure(figsize=(10, 6))
    plot_apple.plot(sequence1, label = 'Apple', linestyle='-', color=color1)
    plot_amazon.plot(sequence2, label = 'Amazon', linestyle='-', color=color2)
    plot_both.plot(sequence1, label = 'Apple', linestyle='-', color=color1)
    plot_both.plot(sequence2, label = 'Amazon', linestyle='-', color=color2)
    
    plt.title(title)
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)
    plt.grid(True)
    plot_both.legend()
    plt.tight_layout()
    plt.show()

def plot_two_timeseries_sequence(df_column1, df_column2, title, x_legend, y_legend, color1, color2):
    sequence1 = model.Timeseries(model.TimeseriesArrayStream(df_column1)).to_sequence()
    sequence2 = model.Timeseries(model.TimeseriesArrayStream(df_column2)).to_sequence()
    plot_two_timeseries(sequence1, sequence2, title, x_legend, y_legend, color1, color2)

#conversion from time series to graph

def sequence_to_graph(column, color):
    strategy = TimeseriesToGraphStrategy(
        visibility_constraints=[TimeseriesEdgeVisibilityConstraintsNatural()],
        graph_type="undirected",
        edge_weighting_strategy=EdgeWeightingStrategyNull(),
    )

    g = strategy.to_graph(model.TimeseriesArrayStream(column))


    pos=nx.spring_layout(g.graph, seed=1)
    nx.draw(g.graph, pos, node_size=40, node_color=color)

#this function is used for connecting similar nodes based on oscillation and seasonality
def code_oscillation(graph, nodes, period):
    for node in nodes:
        graph.add_edge(node, node + period)

#this function parses the informations from xml file and generates time series from them
def time_series_from_xml_file(file_path):

    tree = ET.parse(file_path)
    root = tree.getroot()

    financial_statements = root.find('FinancialStatements')
    COAMap = financial_statements.find('COAMap')
    interim_periods = financial_statements.find('InterimPeriods')
    annual_periods = financial_statements.find("AnnualPeriods")
    
    i = 0
    for item in COAMap:
        name = item.get('coaItem')

        elements = annual_periods.findall(f".//lineItem[@coaCode = '{name}']")
        elements2 = interim_periods.findall(f".//lineItem[@coaCode = '{name}']")
        column = []
        column2 = []

        for element in elements:
            column.append(float(element.text))
        
        for element in elements2:
            column2.append(float(element.text))

        title = item.text + " annual"
        title2 = item.text + " interim"
        if i%20 == 0:
            plot_timeseries_sequence(column, title, 'Date', 'Value', "black")
            plot_timeseries_sequence(column2, title2, 'Date', 'Value', "black")
        i += 1

#creating graph using smaller graphs gained with sliding window

#This function creates smaller graphs that are used as nodes for bigger graph
def return_graph(segment):
    strategy = TimeseriesToGraphStrategy(
        visibility_constraints=[TimeseriesEdgeVisibilityConstraintsNatural()],
        graph_type="undirected",
        edge_weighting_strategy=EdgeWeightingStrategyNull(),
    )

    g = strategy.to_graph(model.TimeseriesArrayStream(segment))

    for i in range(len(g.graph.nodes)):
        old_value = g.graph.nodes[i]['value']
        new_value = [old_value]
        g.graph.nodes[i]['value'] = new_value
    
    return g.graph

#This function combines two nodes in a graph.
def combine_nodes(graph, node_1, node_2):
    
    for i in range(len(list(node_1.nodes(data=True)))):
        for j in range(len(node_2.nodes[i]['value'])):
            node_1.nodes[i]['value'].append(node_2.nodes[i]['value'][j])
    
    for neighbor in list(graph.neighbors(node_2)):
        graph.add_edge(node_1, neighbor)

    graph.remove_node(node_2)
    return graph

#This function when called creates a graph which nodes are smaller graphs gained from segments of original data 
#using sliding window. It's nodes (smaller graphs) that are identical are combined into one node.
def to_slidingWindowGraph(data, color, sliding_win_len, column_name):
    
    g = nx.MultiGraph()
    node_prev = None

    for i in range(len(data[column_name]) - sliding_win_len):
        segment = data[i:i + sliding_win_len]
        h = return_graph(segment[column_name])
        g.add_node(h)
        if(node_prev != None):
            g.add_edge(node_prev, h)
        node_prev = h
    
    
    combined = []


    for i, node_1 in enumerate(list(g.nodes)):
        if node_1 not in g:
            continue

        for node_2 in list(g.nodes)[i+1:]:
            if node_2 == None:
                break

            if node_2 not in g:
                continue

            if(set(list(node_1.edges)) == set(list(node_2.edges))):
                g = combine_nodes(g, node_1, node_2)
                if node_1 not in combined:
                    combined.append(node_1)
    
    colors = []

    #If you want to have nodes that were combined be colored differently, just change one of the colors below.
    for node in list(g.nodes):
        if(node in combined):
            colors.append("blue")
        else:
            colors.append(color)

    pos=nx.spring_layout(g, seed=1)
    nx.draw(g, pos, node_size=40, node_color=colors)
    plt.show()

#These are smaller functions to support ones below:

#finds node with lowest index
def min_node(neighbors, nodes):
    ind_min = 2**15
    node_min = None

    for node in neighbors:
        if node in nodes:
            ind = nodes.index(node)
            if ind < ind_min:
                ind_min = ind
                node_min = node
    
    return node_min

#compares two graphs
def is_equal(graph_1, graph_2):
    if(graph_1.nodes != graph_2.nodes): return False
    if(graph_1.edges != graph_2.edges): return False
    for i in range(len(graph_1.nodes)):
        if list(graph_1.nodes(data=True)[i]['value']) != list(graph_2.nodes(data=True)[i]['value']):
                return False
    return True

#appends values of a node(smaller graphs) by choosing the lowest index available
def append_one(sequence, graph, dict, i):

    dict = {}

    for i in range(len(list(graph.nodes))):
        dict[i] = 0

    if int(dict[i]/2) >= len(list(graph.nodes(data=True)[0]['value'])):
        dict[i] = 0
    
    index = int(dict[i]/2)

    
    for node in graph.nodes(data=True):
        sequence.append(node[1]['value'][index])
    
    dict[i] += 1
    return sequence

#appends values of a node(smaller graphs) randomly

class choose_strategy_2():
    def __init__(self, walk, walkthrough, value, dict, graph, nodes_1, nodes_2):
        self.walkthrough = walkthrough
        self.walk = walk
        self.value = value
        self.dict = dict
        self.graph = graph
        self.nodes_1 = nodes_1
        self.nodes_2 = nodes_2

    def append_random(self, sequence, graph):

        nodes = list(graph.nodes(data = True))
        random.shuffle(nodes)

        for node in nodes:
            index = random.randint(0, len(node[1]['value']) - 1)
            sequence.append(node[1]['value'][index])
        return sequence
    
    def append_lowInd(self, sequence, graph, i):
        if int(self.dict[i]/2) >= len(list(graph.nodes(data=True)[0]['value'])):
            self.dict[i] = 0
    
        index = int(self.dict[i]/2)

        for node in graph.nodes(data=True):
            sequence.append(node[1]['value'][index])
    
        self.dict[i] += 1
        i += 1
        return sequence
    
    def append(self, sequence, graph, i):
        if(self.value) == "random":
            return self.append_random(sequence, graph)
        elif self.value == "sequential" :
            return self.append_lowInd(sequence, graph, i)
        else:
            print("you chose non-existent method of value selection")
            print("please choose between: random, sequential")
            return None

    def next_node_one_random(self, graph_index, node):
        neighbors = set(self.graph.neighbors(node))
        if graph_index == 1:
            neighbors_1 = list(set(self.nodes_1) & neighbors)
            return random.choice(neighbors_1)
        else:
            neighbors_2 = list(set(self.nodes_2) & neighbors)
            return random.choice(neighbors_2)
    
    def next_node_both_random(self, i, graph_index, node_1, node_2, switch):
        neighbors = []
        if (i/switch)%2 == 0:
            neighbors = set(self.graph.neighbors(node_1))
        else:
            neighbors = set(self.graph.neighbors(node_2))
        
        if graph_index == 1:
            neighbors_1 = list(set(self.nodes_1) & neighbors)
            return random.choice(neighbors_1)
        else:
            neighbors_2 = list(set(self.nodes_2) & neighbors)
            return random.choice(neighbors_2)

    def next_node_one_weighted(self, graph_index, node):
        neighbors = set(self.graph.neighbors(node))
        if graph_index == 1:
            neighbors = list(set(self.nodes_1) & neighbors)
        else:
            neighbors = list(set(self.nodes_2) & neighbors)
        
        weights = []
        total = 0
        for neighbor in neighbors:
            num = self.graph.number_of_edges(node, neighbor)
            weights.append(num)
            total += num
        for element in weights:
            element /= total
        
        return random.choices(neighbors, weights=weights, k=1)[0]
    
    def next_node_both_weighted(self, i, graph_index, node_1, node_2, switch):
        neighbors = []
        if (i/switch)%2 == 0:
            neighbors = set(self.graph.neighbors(node_1))
        else:
            neighbors = set(self.graph.neighbors(node_2))
        
        if graph_index == 1:
            neighbors = list(set(self.nodes_1) & neighbors)
        else:
            neighbors = list(set(self.nodes_2) & neighbors)
        
        weights = []
        total = 0
        for neighbor in neighbors:
            num = 0
            if (i/switch)%2 == 0:
                num = self.graph.number_of_edges(node_1, neighbor)
            else:
                num = self.graph.number_of_edges(node_2, neighbor)
            weights.append(num)
            total += num
        for element in weights:
            element /= total
        
        return random.choices(neighbors, weights=weights, k=1)[0]
    
    def next_node_one(self, graph_index, node):
        if self.walkthrough == "random":
            return self.next_node_one_random(graph_index, node)
        elif  self.walkthrough == "weighted":
            return self.next_node_one_weighted(graph_index, node)
        else:
            print("you chose non-existent walkthrough.")
            print("please choose between: random, weighted")
            return None
    
    def next_node_both(self, i, graph_index, node_1, node_2, switch):
        if self.walkthrough == "random":
            return self.next_node_both_random(i, graph_index, node_1, node_2, switch)
        elif  self.walkthrough == "weighted":
            return self.next_node_both_weighted(i, graph_index, node_1, node_2, switch)
        else:
            print("you chose non-existent walkthrough.")
            print("please choose between: random, weighted")
            return None

    def next_node(self, i, graph_index, node_1, node_2, switch):
        if self.walk == "one":
            return self.next_node_one(graph_index, node_1)
        elif self.walk == "both":
            return self.next_node_both(i, graph_index, node_1, node_2, switch)
        else:
            print("you chose non-existent walk")
            print("please choose between: one, both")
            return None        
    
class choose_strategy():
    def __init__(self, walkthrough, value, dict, graph, nodes):
        self.walkthrough = walkthrough
        self.value = value
        self.dict = dict
        self.graph = graph
        self.nodes = nodes
    
    def append_random(self, sequence, graph):

        nodes = list(graph.nodes(data = True))
        random.shuffle(nodes)

        for node in nodes:
            index = random.randint(0, len(node[1]['value']) - 1)
            sequence.append(node[1]['value'][index])
        return sequence

    def append_lowInd(self, sequence, graph, i):
        if int(self.dict[i]/2) >= len(list(graph.nodes(data=True)[0]['value'])):
            self.dict[i] = 0
    
        index = int(self.dict[i]/2)

        for node in graph.nodes(data=True):
            sequence.append(node[1]['value'][index])
    
        self.dict[i] += 1
        i += 1
        return sequence

    def append(self, sequence, graph, i):
        if(self.value) == "random":
            return self.append_random(sequence, graph)
        elif self.value == "sequential" :
            return self.append_lowInd(sequence, graph, i)
        else:
            print("you chose non-existent method of value selection")
            print("please choose between: random, sequential")
            return None

    def next_node_weighted(self, node):
        neighbors = set(self.graph.neighbors(node))
        neighbors = list(set(self.nodes) & neighbors)
        
        weights = []
        total = 0
        for neighbor in neighbors:
            num = self.graph.number_of_edges(node, neighbor)
            weights.append(num)
            total += num
        for element in weights:
            element /= total
        
        return random.choices(neighbors, weights=weights, k=1)[0]

    def next_node_random(self, node):
        neighbors = set(self.graph.neighbors(node))
        neighbors = list(set(self.nodes) & neighbors)
        return random.choice(neighbors)

    def next_node(self, node):
        if self.walkthrough == "random":
            return self.next_node_random(node)
        elif  self.walkthrough == "weighted":
            return self.next_node_weighted(node)
        else:
            print("you chose non-existent walkthrough.")
            print("please choose between: random, weighted")
            return None

class choose_strategy_multiple():
    def __init__(self, walk, walkthrough, value, graph, nodes, dictionaries):
        self.walkthrough = walkthrough
        self.walk = walk
        self.value = value
        self.graph = graph
        self.nodes = nodes
        self.dictionaries = dictionaries
    
    def append_random(self, sequence, graph):

        nodes = list(graph.nodes(data = True))
        random.shuffle(nodes)

        for node in nodes:
            index = random.randint(0, len(node[1]['value']) - 1)
            sequence.append(node[1]['value'][index])
        return sequence

    def append_lowInd(self, sequence, graph, graph_index, index):
        
        if int(self.dictionaries[graph_index][index]/2) >= len(list(graph.nodes(data=True)[0]['value'])):
            self.dictionaries[graph_index][index] = 0
    
        ind = int(self.dictionaries[graph_index][index]/2)

        for node in graph.nodes(data=True):
            sequence.append(node[1]['value'][ind])
    
        self.dictionaries[graph_index][index] += 1
        return sequence

    def append(self, sequence, graph, graph_index, index):
        if(self.value) == "random":
            return self.append_random(sequence, graph)
        elif self.value == "sequential" :
           return self.append_lowInd(sequence, graph, graph_index, index)
        else:
            print("you chose non-existent method of value selection")
            print("please choose between: random")
            return None

    def next_node_one_random(self, graph_index, node):
        neighbors = set(self.graph.neighbors(node))
        neighbors = list(set(self.nodes[graph_index]) & neighbors)
        return random.choice(neighbors)

    def next_node_one_weighted(self, graph_index, node):
        neighbors = set(self.graph.neighbors(node))
        neighbors = list(set(self.nodes[graph_index]) & neighbors)
        
        weights = []
        total = 0
        for neighbor in neighbors:
            num = self.graph.number_of_edges(node, neighbor)
            weights.append(num)
            total += num
        for element in weights:
            element /= total
        
        return random.choices(neighbors, weights=weights, k=1)[0]

    def next_node_one(self, graph_index, node):
        if self.walkthrough == "random":
            return self.next_node_one_random(graph_index, node)
        elif  self.walkthrough == "weighted":
            return self.next_node_one_weighted(graph_index, node)
        else:
            print("you chose non-existent walkthrough.")
            print("please choose between: random, weighted")
            return None

    def next_node_both_random(self, i, graph_index, nodes, switch):
        index = int((i/switch) % len(nodes))
        neighbors = set(self.graph.neighbors(nodes[index]))
        
        neighbors = list(set(self.nodes[graph_index]) & neighbors)
        return random.choice(neighbors)
    
    def next_node_all_weighted(self, i, graph_index, nodes, switch):
        
        index = int((i/switch) % len(nodes))
        neighbors = set(self.graph.neighbors(nodes[index]))
        neighbors = list(set(self.nodes[graph_index]) & neighbors)
        
        weights = []
        total = 0

        for neighbor in neighbors:
            num = self.graph.number_of_edges(nodes[index], neighbor)
            weights.append(num)
            total += num
        
        for element in weights:
            element /= total
        
        return random.choices(neighbors, weights=weights, k=1)[0]
    
    def next_node_all(self, i, graph_index, nodes, switch):
        if self.walkthrough == "random":
            return self.next_node_all_random(i, graph_index, nodes, switch)
        elif  self.walkthrough == "weighted":
            return self.next_node_all_weighted(i, graph_index, nodes, switch)
        else:
            print("you chose non-existent walkthrough.")
            print("please choose between: random, weighted")
            return None
    
    def next_node(self, i, graph_index, nodes, switch):
        if self.walk == "one":
            return self.next_node_one(graph_index, nodes[0])
        elif self.walk == "all":
            return self.next_node_all(i, graph_index, nodes, switch)
        else:
            print("you chose non-existent walk")
            print("please choose between: one, all")
            return None

#--------------------------------------------------

#This function draws a graph made with sliding window mechanism by iterating through the first graph, 
#always choosing lowest available index of a node and value. It simultaniously deletes paths it has already crossed.
def toTimeSequence_lowestIndex(data_1, data_2, sliding_win_len, column_name, color_1 = "red", color_2 = "blue", time_series_len = 100):
    
    graph, nodes_1, nodes_2 = middle_man(data_1, data_2, sliding_win_len, column_name)
    
    sequence_1 = []
    sequence_2 =[]

    current_node_1 = nodes_1[0]
    current_node_2 = nodes_2[0]
    
    previous_node_1 = current_node_1
    previous_node_2 = current_node_2

    dict = {}
    for i in range(len(list(graph.nodes))):
        dict[i] = 0
    
    

    while len(sequence_1) < time_series_len:
        
        previous_node_1 = current_node_1

        index = 0
        for i in range(len(list(graph.nodes))):
            if(is_equal(current_node_1, list(graph.nodes)[i])):
                index = i
                break

        sequence_1 = append_one(sequence_1, current_node_1, dict, index)

        neighbors = list(graph.neighbors(current_node_1))

        current_node_2 = min_node(neighbors, nodes_2)

        index = 0
        for i in range(len(list(graph.nodes))):
            if(is_equal(current_node_2, list(graph.nodes)[i])):
                index = i
                break
        
        sequence_2 = append_one(sequence_2, current_node_2, dict, index)

        if (previous_node_2, current_node_2) in graph.edges:
            graph.remove_edge(previous_node_2, current_node_2)
        
        previous_node_2 = current_node_2

        if neighbors:
            node_min = min_node(neighbors, nodes_1)
        
            if node_min == None:
                print("Node in first graph has no more neighbors")
                break
            
            current_node_1 = node_min
            if (previous_node_1, current_node_1) in graph.edges:
                graph.remove_edge(previous_node_1, current_node_1)

        else:
            print("you have reached the end of this graph")
            break
        
    
    
    plot_timeseries(sequence_1, "graph 1", "Date", "Value", color_1)
    plot_timeseries(sequence_2, "graph 2", "Date", "Value", color_2)

#This function draws a graph made with sliding window mechanism by randomly choosing next node and value always from the same graph, 
# and for the other it chooses a random neighbor of the current node from the first graph
def toTimeSequence_lowestIndex_random(data_1, data_2, sliding_win_len, column_name, color_1 = "red", color_2 = "blue", time_series_len = 100):
    
    graph, nodes_1, nodes_2 = middle_man(data_1, data_2, sliding_win_len, column_name)
    
    sequence_1 = []
    sequence_2 =[]

    current_node_1 = nodes_1[0]
    current_node_2 = nodes_2[0]

    i = 0
    while len(sequence_1) < time_series_len:
        sequence_1 = append_random(sequence_1, current_node_1)
        sequence_2 = append_random(sequence_2, current_node_2)

        neighbors = set(graph.neighbors(current_node_1))
        
        neighbors_1 = list(set(nodes_1) & neighbors)
        neighbors_2 = list(set(nodes_2) & neighbors)

        current_node_1 = random.choice(neighbors_1)
        current_node_2 = random.choice(neighbors_2)
        i += 1
    
    plot_timeseries(sequence_1, "graph 1", "Date", "Value", color_1)
    plot_timeseries(sequence_2, "graph 2", "Date", "Value", color_2)

#This function draws a graph made with sliding window mechanism by randomly choosing next node and value once from fist graph and once from the other
def toTimeSequence_random(data_1, data_2, sliding_win_len, column_name, color_1 = "red", color_2 = "blue", time_series_len = 100):

    
    graph, nodes_1, nodes_2 = middle_man(data_1, data_2, sliding_win_len, column_name)

    sequence_1 = []
    sequence_2 =[]

    current_node_1 = nodes_1[0]
    current_node_2 = nodes_2[0]

    i = 0
    while len(sequence_1) < time_series_len:
        sequence_1 = append_random(sequence_1, current_node_1)
        sequence_2 = append_random(sequence_2, current_node_2)

        neighbors = []
        if i%2 == 0:
            neighbors = set(graph.neighbors(current_node_1))
        else :
            neighbors = set(graph.neighbors(current_node_2))
        
        neighbors_1 = list(set(nodes_1) & neighbors)
        neighbors_2 = list(set(nodes_2) & neighbors)

        current_node_1 = random.choice(neighbors_1)
        current_node_2 = random.choice(neighbors_2)
        i += 1
    
    plot_timeseries(sequence_1, "graph 1", "Date", "Value", color_1)
    plot_timeseries(sequence_2, "graph 2", "Date", "Value", color_2)

#Functions above were integrated into one function belove
#---------------------------------------------------

def toTimeSequence_2(data_1, data_2, sliding_win_len, column_name, walk = "one", walkthrough = "random", select_value = "random", 
    color_1 = "red", color_2 = "blue", time_series_len = 100, skip_values = 1, switch_graphs = 1):

    graph, nodes_1, nodes_2 = middle_man_2(data_1, data_2, sliding_win_len, column_name)

    sequence_1 = []
    sequence_2 =[]

    current_node_1 = nodes_1[0]
    current_node_2 = nodes_2[0]

    dict = {}
    for i in range(len(list(graph.nodes))):
        dict[i] = 0
    
    strategy = choose_strategy_2(walk, walkthrough, select_value, dict, graph, nodes_1, nodes_2)

    i = 0
    while len(sequence_1) < time_series_len:

        index = 0
        for i in range(len(list(graph.nodes))):
            if(is_equal(current_node_1, list(graph.nodes)[i])):
                index = i
                break
        
        sequence_1 = strategy.append(sequence_1, current_node_1, index)
        sequence_2 = strategy.append(sequence_2, current_node_2, index)
        if sequence_1[-1] == None or sequence_2[-1] == None:
            return

        for j in range(skip_values):
            current_node_1 = strategy.next_node(i, 1, current_node_1, current_node_2, switch_graphs)
            current_node_2 = strategy.next_node(i, 2, current_node_1, current_node_2, switch_graphs)
            if(current_node_1 == None or current_node_2 == None):
                return
        i += 1
    
    plot_timeseries(sequence_1, f"walk = {walk}, walkthrough = {walkthrough}, value = {select_value}", "Date", "Value", color_1)
    plot_timeseries(sequence_2, f"walk = {walk}, walkthrough = {walkthrough}, value = {select_value}", "Date", "Value", color_2)

def toTimeSequence(data, sliding_win_len, column, color = "black", walkthrough = "random", 
                                select_value = "random", time_series_len = 100, skip_values = 1):
    
    graph = middle_man(data, sliding_win_len, column)
    sequence = []
    nodes = list(graph.nodes)
    current_node = nodes[0]

    dict = {}
    for i in range(len(list(graph.nodes))):
        dict[i] = 0

    strategy = choose_strategy(walkthrough, select_value, dict, graph, nodes)

    while len(sequence) < time_series_len:

        index = 0
        for i in range(len(list(graph.nodes))):
            if(is_equal(current_node, list(graph.nodes)[i])):
                index = i
                break
        
        sequence = strategy.append(sequence, current_node, index)
        if sequence[-1] == None:
            return

        for j in range(skip_values):
            current_node = strategy.next_node(current_node)
            if(current_node == None):
                return

    plot_timeseries(sequence, f"walkthrough = {walkthrough}, value = {select_value}", "Date", "Value", color)

def toMultipleTimeSequences(data, sliding_win_len, column, colors = None, walk = "one", walkthrough = "random", 
                            select_value = "random", time_series_len = 100, skip_values = 1, switch_graphs = 1):
    
    graph, nodes = middle_man_multiple(data, sliding_win_len, column)
    sequences = [[] for _ in range(len(data))]

    current_nodes = [None for _ in range(len(data))]

    for i in range(len(data)):
        current_nodes[i] = nodes[i][0]
    
    dictionaries = [{} for _ in range(len(data))]
    for i in range(len(data)):
        for j in range(len(list(nodes[i]))):
            dictionaries[i][j] = 0

    
    
    strategy = choose_strategy_multiple(walk, walkthrough, select_value, graph, nodes, dictionaries)

    i = 0
    while len(sequences[0]) < time_series_len:
        
        for j in range(len(sequences)):

            index = 0
            for i in range(len(list(nodes[j]))):
                if(is_equal(current_nodes[j], list(graph.nodes)[i])):
                    index = i
                    break

            sequences[j] = strategy.append(sequences[j], current_nodes[j], j, index)
            if sequences[j][-1] == None:
                return

        for j in range(skip_values):
            
            for k in range(len(current_nodes)):
                current_nodes[k] = strategy.next_node(i, k, current_nodes, switch_graphs)
                if(current_nodes[k] == None):
                    return
        i += 1
    
    if colors == None:
        for j in range(len(sequences)):
            colors[j] = "black"
    
    for j in range(len(sequences)):
        plot_timeseries(sequences[j], f"walk = {walk}, walkthrough = {walkthrough}, value = {select_value}", "Date", "Value", colors[j])



#This functions binds two graphs made by the method of sliding window based on time co-ocurrance 
#and then combines identical nodes(smaller graphs).
#At the end you can add a call to the function to draw these graphs either randomly or iterating through one
def connect_sliding_win_graphs(data_1, data_2, sliding_win_len, column_name, color_1 = "red", color_2 = "blue", color_3 = "green", color_4 = "orange"):
    
    g, nodes_1, nodes_2 = middle_man(data_1, data_2, sliding_win_len, column_name)

    combined_1 = []
    combined_2 = []


    for i, node_1 in enumerate(list(nodes_1)):
        if node_1 not in g:
            continue

        for node_2 in list(nodes_1)[i+1:]:
            if node_2 == None:
                continue

            if node_2 not in g:
                continue

            if(set(list(node_1.edges)) == set(list(node_2.edges))):
                g = combine_nodes(g, node_1, node_2)
                if node_1 not in combined_1:
                    combined_1.append(node_1)

    
    for i, node_1 in enumerate(list(nodes_2)):
        if node_2 == None:
                continue
            
        if node_1 not in g:
            continue

        for node_2 in list(nodes_2)[i+1:]:
            if node_2 not in g:
                continue
        
            if(set(list(node_1.edges)) == set(list(node_2.edges))):
                g = combine_nodes(g, node_1, node_2)
                if node_1 not in combined_2:
                    combined_2.append(node_1)


    colors = []

    #If you want to have nodes that were combined be colored differently, just change one of the colors below.
    for node in list(g.nodes):
        if(node in combined_1):
            colors.append(color_1)
        elif(node in combined_2):
            colors.append(color_2)
        elif node in nodes_1 :
            colors.append(color_3)
        else:
            colors.append(color_4)
    

    pos=nx.spring_layout(g, seed=1)
    nx.draw(g, pos, node_size=40, node_color=colors)
    plt.show()
    
    #toTimeSequence_lowestIndex(g, nodes_1, nodes_2, "red", "blue", time_series_len=200)
    #toTimeSequence_lowestIndex_random(g, nodes_1, nodes_2, "green", "black", time_series_len=200)
    #toTimeSequence_random(g, nodes_1, nodes_2, "grey", "purple", time_series_len=200)

def middle_man(data, sliding_win_len, column):
    g = nx.MultiGraph()
    node_prev = None

    for i in range(len(data) - sliding_win_len):
        segment = data[i:i + sliding_win_len][column]
        h = return_graph(segment)
        g.add_node(h)
        if(node_prev != None):
            g.add_edge(node_prev, h)
        node_prev = h
    
    return g

def middle_man_2(data_1, data_2, sliding_win_len, column_name):
    g = nx.MultiGraph()

    node_prev = None

    nodes_1 = []
    nodes_2 = []

    for i in range(len(data_1) - sliding_win_len):
        segment = data_1[i:i + sliding_win_len][column_name]
        h = return_graph(segment)
        g.add_node(h)
        if(node_prev != None):
            g.add_edge(node_prev, h)
        node_prev = h
        nodes_1.append(h)

    for i in range(len(data_2) - sliding_win_len):
        segment = data_2[i:i + sliding_win_len][column_name]
        h = return_graph(segment)
        g.add_node(h)
        if(node_prev != None):
            g.add_edge(node_prev, h)
        node_prev = h
        nodes_2.append(h)
    
    
    for i in range(0, len(nodes_1)):
        g.add_edge(nodes_1[i], nodes_2[i])
    
    return g, nodes_1, nodes_2

def middle_man_multiple(data, sliding_win_len, column):
    g = nx.MultiGraph()

    node_prev = None
    nodes = [[] for _ in range(len(data))]

    for i in range(len(nodes)):
        node_prev = None
        for j in range(len(data[i]) - sliding_win_len):
            segment = data[i][j:j + sliding_win_len][column]
            h = return_graph(segment)
            g.add_node(h)
            if(node_prev != None):
                g.add_edge(node_prev, h)
            node_prev = h
            nodes[i].append(h)
    
    for i in range(len(data)):
        for j in range(len(data)):
            if i == j:
                continue
            for k in range(len(nodes[i])):
                g.add_edge(nodes[i][k], nodes[j][k])
    
    return g, nodes

#connect_sliding_win_graphs(segment_amazon, segment_apple, 5, "Close")

#toTimeSequence_2(segment_amazon, segment_apple, 5, "Close", walk = "both", walkthrough="weighted", select_value="sequential", skip_values = 5, time_series_len=200, switch_graphs=6)
#toTimeSequence_2(segment_amazon, segment_apple, 5, "Close", walk = "both", walkthrough="weighted")
#toTimeSequence_2(segment_amazon, segment_apple, 5, "Close", walk = "both", select_value="sequential")
#toTimeSequence_2(segment_amazon, segment_apple, 5, "Close", walk = "both")
#toTimeSequence_2(segment_amazon, segment_apple, 5, "Close", walkthrough="weighted", select_value="sequential")
#toTimeSequence_2(segment_amazon, segment_apple, 5, "Close", walkthrough="weighted")
#toTimeSequence_2(segment_amazon, segment_apple, 5, "Close", walkthrough="weighted", select_value="sequential")
#toTimeSequence_2(segment_amazon, segment_apple, 5, "Close")
"""
data = [segment_amazon, segment_apple, segment_amazon, segment_apple]
colors = ["green", "orange", "blue", "pink"]
toMultipleTimeSequences(data, 5, "Close", colors, "all", "weighted", "sequential", 200, 5, 10)
toTimeSequence(segment_amazon, 5, "Close", color = "blue", walkthrough= "weighted", select_value="sequential", time_series_len=200, skip_values=5)
toTimeSequence(segment_apple, 5, "Close", color = "red", walkthrough= "weighted", select_value="sequential", time_series_len=200, skip_values=5)
"""
to_slidingWindowGraph(segment_apple, "pink", 5, "Close")
sequence_to_graph(segment_apple["Close"], "purple")
plt.show()