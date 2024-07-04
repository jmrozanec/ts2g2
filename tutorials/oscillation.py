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


#this function connects the nodes that are peaks in their respective oscillations. 
#difference in value between peaks must be less than osc_all_dif for them to be considered to be in same oscillation
#same applies for troughs
def sequence_to_graph_with_small_oscillation(column, color, osc_all_dif):
    strategy = TimeseriesToGraphStrategy(
        visibility_constraints=[TimeseriesEdgeVisibilityConstraintsNatural()],
        graph_type="undirected",
        edge_weighting_strategy=EdgeWeightingStrategyNull(),
    )

    peaks, peak_dict = find_peaks(column)
    troughs, trough_dict = find_peaks(-column)

    g = strategy.to_graph(model.TimeseriesArrayStream(column))

    for node in peaks:
        for node2 in peaks:
            if node2 <= node:
                continue
            value = g.graph.nodes[node]['value']
            value2 = g.graph.nodes[node2]['value']
            dif = abs(value - value2)

            if dif > osc_all_dif:
                break
            g.graph.add_edge(node, node2)
    
    for node in troughs:
        for node2 in troughs:
            if node2 <= node:
                continue
            value = g.graph.nodes[node]['value']
            value2 = g.graph.nodes[node2]['value']
            dif = abs(value - value2)
            if dif > osc_all_dif:
                break
            g.graph.add_edge(node, node2)
    
    pos=nx.spring_layout(g.graph, seed=1)
    nx.draw(g.graph, pos, node_size=40, node_color=color)
    plt.show()
    

def sequence_to_graph(column, color):
    strategy = TimeseriesToGraphStrategy(
        visibility_constraints=[TimeseriesEdgeVisibilityConstraintsNatural()],
        graph_type="undirected",
        edge_weighting_strategy=EdgeWeightingStrategyNull(),
    )

    g = strategy.to_graph(model.TimeseriesArrayStream(column))
    
    pos=nx.spring_layout(g.graph, seed=1)
    nx.draw(g.graph, pos, node_size=40, node_color=color)
    plt.show()
    



#this function connects first peaks in each oscillation provided that their value is at least min_dif and at most max_dif apart.
#consecutive peaks are considered to be in same oscillation if their values are at most min_dif apart.
#same applies for troughs
def sequence_to_graph_with_big_oscillation(column, color, min_dif, max_dif):
    strategy = TimeseriesToGraphStrategy(
        visibility_constraints=[TimeseriesEdgeVisibilityConstraintsNatural()],
        graph_type="undirected",
        edge_weighting_strategy=EdgeWeightingStrategyNull(),
    )

    peaks, peak_dict = find_peaks(column)
    troughs, trough_dict = find_peaks(-column)

    g = strategy.to_graph(model.TimeseriesArrayStream(column))

    last_node = peaks[0]
    node_bank = peaks[0]
    for node in peaks:
        if node < last_node:
            continue

        for node2 in peaks:
            if node2 < node:
                continue
            
            value = g.graph.nodes[node]['value']
            value2 = g.graph.nodes[node2]['value']
            dif1 = abs(value - value2)
            value_bank = g.graph.nodes[node_bank]['value']
            dif2 = abs(value2 - value_bank)

            if dif1 < min_dif or dif1 > max_dif:
                continue
            if dif2 < min_dif:
                continue

            g.graph.add_edge(node, node2)

            node_bank = node2

            if node == last_node:
                last_node = node2
    
    last_node = troughs[0]
    node_bank = troughs[0]
    for node in troughs:
        if node < last_node:
            continue

        for node2 in troughs:
            if node2 < node:
                continue
            
            value = g.graph.nodes[node]['value']
            value2 = g.graph.nodes[node2]['value']
            dif1 = abs(value - value2)
            value_bank = g.graph.nodes[node_bank]['value']
            dif2 = abs(value2 - value_bank)

            if dif1 < min_dif or dif1 > max_dif:
                continue
            if dif2 < min_dif:
                continue

            g.graph.add_edge(node, node2)

            node_bank = node2

            if node == last_node:
                last_node = node2

    pos=nx.spring_layout(g.graph, seed=1)
    nx.draw(g.graph, pos, node_size=40, node_color=color)
    plt.show()




plot_timeseries_sequence(segment_amazon["Close"], "segment", "Date", "Value", "orange")
sequence_to_graph(segment_amazon["Close"], "red")
sequence_to_graph_with_small_oscillation(segment_amazon["Close"], "blue", 20)
sequence_to_graph_with_big_oscillation(segment_amazon["Close"], "pink", 16, 500)