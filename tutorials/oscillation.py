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

from scipy.signal import find_peaks, welch


#data
apple_data = pd.read_csv(os.path.join(os.getcwd(), "apple", "APPLE.csv"))

apple_data["Date"] = pd.to_datetime(apple_data["Date"])
apple_data.set_index("Date", inplace=True)

amazon_data = pd.read_csv(os.path.join(os.getcwd(), "amazon", "AMZN.csv"))

amazon_data["Date"] = pd.to_datetime(amazon_data["Date"])
amazon_data.set_index("Date", inplace=True)

segment_apple = apple_data[60:720]
segment_amazon = amazon_data[5629:6289]
segment_1 = apple_data[60:1200]


#drawing time series
def plot_timeseries(sequence, title, x_legend, y_legend, color):
    plt.figure(figsize=(10, 6))
    plt.plot(sequence, linestyle='-', color=color)
    
    plt.title(title)
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)
    plt.grid(True)
    plt.show()


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
    return g.graph

#This function combines two nodes in a graph.
def combine_nodes(graph, node_1, node_2):
    
    for neighbor in list(graph.neighbors(node_2)):
        graph.add_edge(node_1, neighbor)

    graph.remove_node(node_2)
    #return graph



#This function when called creates a graph which nodes are smaller graphs gained from segments of original data 
#using sliding window. It's nodes (smaller graphs) that are identical are combined into one node.
def sliding_win_graph(data, color, sliding_win_len, column_name):
    
    g = nx.Graph()
    node_prev = None

    for i in range(len(data) - sliding_win_len):
        segment = data[i:i + sliding_win_len][column_name]
        h = return_graph(segment)
        g.add_node(h)
        if(node_prev != None):
            g.add_edge(node_prev, h)
        node_prev = h
    
    combined = []

    for i, node_1 in enumerate(list(g.nodes)):
        for node_2 in list(g.nodes)[i:]:
            
            if(set(list(node_1.edges)) == set(list(node_2.edges))):
                combine_nodes(g, node_1, node_2)
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

    
#sliding_win_graph(segment_apple, "red", 10, "Close")
#sliding_win_graph(segment_1, "green", 10, "Close")
#sliding_win_graph(segment_amazon, "pink", 5, "Close")
plt.show()

import xml.etree.ElementTree as ET

#this function parses the informations from xml file and generates time series from them
def time_series_from_xml_file():
    file_path = os.path.join(r'C:\Users\Žan Grčar\python\ts2g2\fundimentials', r'0198N\ReportsFinStatements.xml')
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


#time_series_from_xml_file()

#This functions binds two graphs made by the method of sliding window based on time co-ocurrance 
#and then combines identical nodes(smaller graphs).
def connected_sliding_win_graphs(data_1, data_2, color, sliding_win_len, column_name):

    g = nx.Graph()
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

    nodes_1_copy = nodes_1
    nodes_2_copy = nodes_2

    for i, node_1 in enumerate(list(nodes_1)):
        for node_2 in list(nodes_1)[i:]:
            
            if(set(list(node_1.edges)) == set(list(node_2.edges))):
                combine_nodes(g, node_1, node_2)
                combined_1.append(node_1)
                nodes_1.remove(node_2)
    
    for i, node_1 in enumerate(list(nodes_2)):
        for node_2 in list(nodes_2)[i:]:
            
            if(set(list(node_1.edges)) == set(list(node_2.edges))):
                combine_nodes(g, node_1, node_2)
                combined_2.append(node_1)
                nodes_2.remove(node_2)
    
    colors = []

    #If you want to have nodes that were combined be colored differently, just change one of the colors below.
    for node in list(g.nodes):
        if(node in combined_1):
            colors.append("blue")
        elif(node in combined_2):
            colors.append("red")
        elif node in nodes_1_copy :
            colors.append("green")
        else:
            colors.append("violet")

    pos=nx.spring_layout(g, seed=1)
    nx.draw(g, pos, node_size=40, node_color=colors)
    plt.show()
    
connected_sliding_win_graphs(segment_amazon, segment_apple, "red", 5, "Close")
#connected_sliding_win_graphs(segment_amazon, segment_apple, "red", 10, "Close")

plt.show()