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
def sliding_win_graph(data, color, sliding_win_len, column_name):
    
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

    if int(dict[i]/2) >= len(list(graph.nodes(data=True)[0]['value'])):
        dict[i] = 0
    
    index = int(dict[i]/2)

    
    for node in graph.nodes(data=True):
        sequence.append(node[1]['value'][index])
    
    dict[i] += 1
    return sequence

#appends values of a node(smaller graphs) randomly
def append_random(sequence, graph):

    nodes = list(graph.nodes(data = True))
    random.shuffle(nodes)

    for node in nodes:
        index = random.randint(0, len(node[1]['value']) - 1)
        sequence.append(node[1]['value'][index])
    return sequence
#--------------------------------------------------

#This function draws a graph made with sliding window mechanism by iterating through the first graph, 
#always choosing lowest available index of a node and value. It simultaniously deletes paths it has already crossed.
def toTimeSequence_lowestIndex(graph, nodes_1, nodes_2, color_1, color_2, time_series_len = 100):
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
def toTimeSequence_lowestIndex_random(graph, nodes_1, nodes_2, color_1, color_2, time_series_len = 100):
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
def toTimeSequence_random(graph, nodes_1, nodes_2, color_1, color_2, time_series_len = 100):
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

#This functions binds two graphs made by the method of sliding window based on time co-ocurrance 
#and then combines identical nodes(smaller graphs).
#At the end you can add a call to the function to draw these graphs either randomly or iterating through one
def connected_sliding_win_graphs(data_1, data_2, sliding_win_len, column_name):

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
            colors.append("blue")
        elif(node in combined_2):
            colors.append("red")
        elif node in nodes_1 :
            colors.append("green")
        else:
            colors.append("violet")
    

    pos=nx.spring_layout(g, seed=1)
    nx.draw(g, pos, node_size=40, node_color=colors)
    plt.show()

    #here you can choose how you want them to be drawn and how long do you want the sequence to be
    
    #graphToTime_sequence_one_least(g, nodes_1, nodes_2, "red", "blue", time_series_len=200)
    #graphToTime_sequence_one_random(g, nodes_1, nodes_2, "green", "black", time_series_len=200)
    #graphToTime_sequence_random(g, nodes_1, nodes_2, "grey", "purple", time_series_len=200)

connected_sliding_win_graphs(segment_amazon, segment_apple, 5, "Close")

plt.show()