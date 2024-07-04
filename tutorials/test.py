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


#zloadanje podatkov

amazon_data = pd.read_csv(os.path.join(os.getcwd(), "amazon", "AMZN.csv"))

amazon_data["Date"] = pd.to_datetime(amazon_data["Date"])
amazon_data.set_index("Date", inplace=True)


#iz podatkov v Time series

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


plot_timeseries_sequence(amazon_data["Close"], "Original Sequence", "Year", "Value", "blue") # tuki spreminji barvo kokr čš

#po segmentih

segment_1 = amazon_data[60:120]
segment_2 = amazon_data[4000:4060]
segment_3 = amazon_data[6000:6060]

plot_timeseries_sequence(segment_1["Close"], "Segment 1", "Year", "Value", "red")
plot_timeseries_sequence(segment_2["Close"], "Segment 2", "Year", "Value", "green")
plot_timeseries_sequence(segment_3["Close"], "Segment 3", "Year", "Value", "brown")


#zdj pa iz timeseries v grafe

def sequence_to_graph(column, color):
    strategy = TimeseriesToGraphStrategy(
        visibility_constraints=[TimeseriesEdgeVisibilityConstraintsNatural()],

        #tuki mamo lohk vec načinov izrisevanja grafov:
        #TimeseriesEdgeVisibilityConstraintsNatural
        #TimeseriesEdgeVisibilityConstraintsHorizontal
        #TimeseriesEdgeVisibilityConstraintsVisibilityAngle

        graph_type = "undirected",
        edge_weighting_strategy=EdgeWeightingStrategyNull()

    )

    g = strategy.to_graph(model.TimeseriesArrayStream(column))
    pos = nx.spring_layout(g.graph, seed=1)
    nx.draw(g.graph, pos, node_size = 40, node_color=color)


sequence_to_graph(segment_1["Close"], "gray")
sequence_to_graph(segment_2["Close"], "yellow")
sequence_to_graph(segment_3["Close"], "purple")



#zdej pa še iz grafa v timesequence


def generate_sequence(df, reference_column, color):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=100)
    stream_train = model.TimeseriesArrayStream(df_train[reference_column])

    strategy = TimeseriesToGraphStrategy(
        visibility_constraints=[TimeseriesEdgeVisibilityConstraintsNatural()],
        graph_type="undirected",
        edge_weighting_strategy=EdgeWeightingStrategyNull(),
    )
    graph = strategy.to_graph(stream_train)
    test_length = len(df_test.index)
    sequence = graph.to_sequence(RandomWalkWithRestartSequenceGenerationStrategy(), sequence_length=test_length)
    
    dfx = pd.DataFrame(
    {'y_true': df[reference_column].values,
     'y_pred': [None]*len(df_train.index)+sequence
    })
    dfx.index = df.index
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(dfx['y_true'], linestyle='-.', color=color)
    plt.plot(dfx['y_pred'], linestyle='-', color='black')
    plt.title("Random Walk With Restart Sequence Generated Strategy")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(False)
    plt.show()


generate_sequence(segment_1, 'Close', 'gray')
generate_sequence(segment_2, 'Close', 'green')
generate_sequence(segment_3, 'Close', 'blue')





