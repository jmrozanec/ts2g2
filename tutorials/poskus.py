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

apple_data = pd.read_csv(os.path.join(os.getcwd(), "apple", "APPLE.csv"))

apple_data["Date"] = pd.to_datetime(apple_data["Date"])
apple_data.set_index("Date", inplace=True)

amazon_data = pd.read_csv(os.path.join(os.getcwd(), "amazon", "AMZN.csv"))

amazon_data["Date"] = pd.to_datetime(amazon_data["Date"])
amazon_data.set_index("Date", inplace=True)


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


segment_apple = apple_data[60:120]
segment_amazon = amazon_data[5629:5689]
segment_1 = amazon_data[4500:5200]

#plot_timeseries_sequence(amazon_data["Close"], "Amazon", "Year", "Value", "black")
#plot_timeseries_sequence(apple_data["Close"], "Apple", "Year", "Value", "grey")
plot_timeseries_sequence(segment_1["Close"], "Apple", "Year", "Value", "silver")

def group_graph(column1, color1, column2, color2):
    strategy = TimeseriesToGraphStrategy(
        visibility_constraints=[TimeseriesEdgeVisibilityConstraintsNatural()],
        graph_type = "undirected",
        edge_weighting_strategy=EdgeWeightingStrategyNull()

    )

    g = strategy.to_graph(model.TimeseriesArrayStream(column1))
    f = strategy.to_graph(model.TimeseriesArrayStream(column2))
    h = nx.union(g.graph, f.graph, rename=("g", "f"))

    color_map = []
    
    h_nodes = list(h.nodes)

    for node in h_nodes:
        if node < "g":
            color_map.append(color1)
        else :
            color_map.append(color2)
    
    for gn, fn in zip(g.graph.nodes, f.graph.nodes):
        g_idx = h_nodes.index(f"g{gn}")
        f_idx = h_nodes.index(f"f{fn}")
        value_g = g.graph.nodes[gn]['value']
        value_f = f.graph.nodes[fn]['value']
        dif = 0
        if value_g > value_f:
            dif = value_g - value_f
        else:
            dif= value_f - value_g

        h.add_edge(f"g{gn}", f"f{fn}", weight = dif)

    pos = nx.spring_layout(h, seed=1)
    nx.draw(h, pos, node_size = 40, node_color=color_map)
    
    plt.show()
    #print(list(h.edges))


#group_graph(segment_apple["Close"], "blue", segment_amazon["Close"], "red")


#plot_timeseries_sequence(segment_apple["Close"], "Segment 1", "Year", "Value", "grey")
#plot_timeseries_sequence(segment_amazon["Close"], "Segment 1", "Year", "Value", "orange")

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

#plot_two_timeseries_sequence(segment_apple["Close"], segment_amazon["Close"], "Segment 1", "Year", "Value", "grey", "orange")
#plot_two_timeseries_sequence(apple_data["Close"], amazon_data["Close"], "Segment 1", "Year", "Value", "black", "green")