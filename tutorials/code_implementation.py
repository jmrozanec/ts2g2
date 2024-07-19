import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

#import os
import csv
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from timeseries.strategies import TimeseriesToGraphStrategy, TimeseriesEdgeVisibilityConstraintsNatural, TimeseriesEdgeVisibilityConstraintsHorizontal, EdgeWeightingStrategyNull, TimeseriesEdgeVisibilityConstraintsVisibilityAngle
from generation.strategies import RandomWalkWithRestartSequenceGenerationStrategy, RandomWalkSequenceGenerationStrategy, RandomNodeSequenceGenerationStrategy, RandomNodeNeighbourSequenceGenerationStrategy, RandomDegreeNodeSequenceGenerationStrategy
from core import model 
from sklearn.model_selection import train_test_split
import itertools

import xml.etree.ElementTree as ET
import random
import hashlib

class TSprocess:
    """Superclass of classes Segment and SlidingWindow"""
    def __init__(self):
        pass

    def process(time_series):
        pass

class Segment(TSprocess):

    def __init__(self, segment_start, segment_end):
        self.segment_start = segment_start
        self.segment_end = segment_end
    
    def process(self, time_series):
        """returns a TimeSeriesToGraph object, that has a wanted segment of original time serie"""
        return TimeSeriesToGraph(time_series[self.segment_start:self.segment_end])

class SlidingWindow(TSprocess):

    def __init__(self, window_size, win_move_len = 1):
        self.window_size = window_size
        self.win_move_len = win_move_len
    
    def process(self, time_series):
        """Returns a TimeSeriesToGraph object, that has an array of segments of window_size size and each win_move_len data apart.
        This function of this class is called, when we want to create a graph using sliding window mehanism."""
        segments = []
        for i in range(0, len(time_series) - self.window_size, self.win_move_len):
            segments.append(time_series[i:i + self.window_size])
        
        new_series = []
        for i in range(len(segments)):
            new_series.append(TimeSeriesToGraph(segments[i]))
        
        return TimeSeriesToGraph(new_series, True)




class Link:
    def __init__(self, graph = None, multi = False, att = 'value'):
        self.seasonalites = False
        self.same_timestep = -1
        self.graph = graph
        self.coocurrence = False
        self.multi = multi
        self.period = None
        self.attribute = att
    
    def seasonalities(self, period):
        """Notes that we want to connect based on seasonalities, ad sets the period parameter."""
        self.seasonalites = True
        self.period = period
        return self
    
    def same_value(self, allowed_difference):
        """Notes that we want to connect based on similarity of values."""
        self.same_timestep = allowed_difference
        return self

    def time_coocurence(self):
        """Notes that we want to connect graphs in amultivariate graph based on time co-ocurrance."""
        self.coocurrence = True
        return self
    
    def link_positional(self, graph):
        """Connects graphs in a multivariate graph based on time co-ocurrance."""
        g = None
        if self.multi:
            g = nx.MultiGraph()
        else :
            g = nx.Graph()

        min_size = None
        
        for graph in graph.values():
            if min_size == None or len(graph.nodes) < min_size:
                min_size = len(graph.nodes)

        for hash, graph in graph.items():
            nx.set_node_attributes(graph, hash, 'id')
            i = 0
            for node in list(graph.nodes(data = True)):
                node[1]['order'] = i
                i += 1
        
        for graph in graph.values():
            g = nx.compose(g, graph)

        i = 0
        j = 0
        for (node_11, node_12) in zip(list(g.nodes(data = True)), list(g.nodes)):
            
            i = 0
            for (node_21, node_22) in zip(list(g.nodes(data = True)), list(g.nodes)):
                if i == j:
                    i+=1
                    continue

                if node_11[1]['order'] == node_21[1]['order']:
                    g.add_edge(node_12, node_22, intergraph_binding = 'positional')
                i+=1
            j+=1
        
        self.graph = g

    def link_seasonalities(self):
        """Links nodes that are self.period instances apart."""
        for i in range(len(self.graph.nodes) - self.period):
            self.graph.add_edge(list(self.graph.nodes)[i], list(self.graph.nodes)[i+self.period], intergraph_binding='seasonality')

    def link_same_timesteps(self):
        """Links nodes whose values are at most sels.same_timestep apart."""
        for node_11, node_12 in zip(self.graph.nodes(data=True), self.graph.nodes):
            for node_21, node_22 in zip(self.graph.nodes(data=True), self.graph.nodes):
                if  abs(node_11[1][self.attribute][0] - node_21[1][self.attribute][0]) < self.same_timestep and node_12 != node_22:
                    self.graph.add_edge(node_12, node_22, intergraph_binding = 'timesteps')

    def link(self, graph):
        """Calls functions to link nodes based on what we set before."""
        self.graph = graph

        if self.seasonalites:
            self.link_seasonalities()

        if self.same_timestep > 0:
            self.link_same_timesteps()
        
        if self.coocurrence:
            self.link_positional(graph)

        return self.graph




class Strategy:
    """Superclass of classes NaturalVisibility and HorizontalVisibility. Sets and returns a strategy with which we can
    convert time series into graphs."""

    def __init__(self):
        self.visibility = []
        self.graph_type = "undirected"
        self.edge_weighting_strategy = EdgeWeightingStrategyNull()
        self.str_name = None

    def with_angle(self, angle):
        """Sets an angle in which range must a node be to be considered for connection."""
        self.visibility.append(TimeseriesEdgeVisibilityConstraintsVisibilityAngle(angle))
        self.str_name += (f" with angle({angle})")
        return self

    def with_limit(self, limit):
        """Sets a limit as to how many data instances two nodes must be apart to be considered for connection."""
        pass
    
    def strategy_name(self):
        """Returns name of used strategy."""
        return self.str_name

    def get_strategy(self):
        """Returns strategy."""
        return TimeseriesToGraphStrategy(
            visibility_constraints = self.visibility,
            graph_type= self.graph_type,
            edge_weighting_strategy=self.edge_weighting_strategy
        )

class NaturalVisibility(Strategy):
    """As initial strategy sets Natural visibility strategy."""
    def __init__(self):
        super().__init__()
        self.visibility = [TimeseriesEdgeVisibilityConstraintsNatural()]
        self.str_name = "Natural visibility strategy"
    
    def with_limit(self, limit):
        self.visibility[0] = TimeseriesEdgeVisibilityConstraintsNatural(limit)
        self.str_name += (f" with limit({limit})")
        return self
    
class HorizontalVisibility(Strategy):
    """As initial strategy sets Horizontal visibility strategy."""
    def __init__(self):
        super().__init__()
        self.visibility = [TimeseriesEdgeVisibilityConstraintsHorizontal()]
        self.str_name = "Horizontal visibility strategy"
    
    def with_limit(self, limit):
        self.visibility[0] = TimeseriesEdgeVisibilityConstraintsHorizontal(limit)
        self.str_name += (f" with limit({limit})")
        return self




class CsvRead:
    """Superclass of all classes for extraxtion of data from csv files.""" 
    def __init__(self):
        pass

    def from_csv(self):
        pass

class CsvStock(CsvRead):
    """Returns proccessed data from csv file with data sorted by date."""
    def __init__(self, path, y_column):
        self.path = path
        self.y_column = y_column
    
    def from_csv(self):
        time_series = pd.read_csv(self.path)
        time_series["Date"] = pd.to_datetime(time_series["Date"])
        time_series.set_index("Date", inplace=True)
        time_series = time_series[self.y_column]
        return time_series

class XmlRead:
    """Superclass of all classes for extraxtion of data from xml files."""
    def __init__(self):
        pass

    def from_xml(self):
        pass

class XmlSomething(XmlRead):
    """One of the ways of extraction of data from xml file."""
    def __init__(self, path, item, season = "Annual"):
        self.path = path
        self.item = item
        self.season = season
    
    def from_xml(self):
        tree = ET.parse(self.path)
        root = tree.getroot()

        financial_statements = root.find('FinancialStatements')
        COAMap = financial_statements.find('COAMap')
        
        periods = None
        if self.season.lower() == "annual":
            periods = financial_statements.find("AnnualPeriods")
        else:
            periods = financial_statements.find('InterimPeriods')

        elements = periods.findall(f".//lineItem[@coaCode = '{self.item}']")
        column = []

        for element in elements:
            column.append(float(element.text))
        
        return column





class TimeSeriesToGraph():
    
    def __init__(self, time_series = None, slid_win = False, attribute = 'value'):
        self.time_series = time_series
        self.strategy = None
        self.graph = None
        self.orig_graph = None
        self.slid_win = slid_win
        self.slid_graphs = []
        self.attribute = attribute
    
    def from_csv(self, csv_read):
        """Gets data from csv file."""
        self.time_series = csv_read.from_csv()
        return self
    
    def from_xml(self, xml_read):
        """Gets data from xml file."""
        self.time_series = xml_read.from_xml()
        return self

    def return_graph(self):
        """Returns graph."""
        return self.graph

    def return_original_graph(self):
        """Returns graph that still has all of the nodes."""
        if self.orig_graph == None:
            return self.graph
        return self.orig_graph

    def process(self, ts_processing_strategy = None):
        """Returns a TimeSeriesToGraph object that is taylored by ts_processing_strategy. 
        ts_processing_strategy is expected to be an object of a subclass of class TSprocess."""
        if ts_processing_strategy == None:
            return self

        return ts_processing_strategy.process(self.time_series)
        #to do: how to return efficiently

    def to_graph(self, strategy):
        """Converts time serie to graph using strategy strategy. 
        Parameter strategy must be of class Strategy or any of its subclasses."""
        self.strategy = strategy.get_strategy()

        if self.slid_win:
            self.graph = nx.MultiGraph()
            for i in range(len(self.time_series)):
                self.slid_graphs.append(self.time_series[i].to_graph(strategy).return_graph())
            
            for i in range(len(self.slid_graphs)-1):
                self.graph.add_edge(self.slid_graphs[i], self.slid_graphs[i+1])
            
            for graph in self.graph.nodes:
                for i in range(len(graph.nodes)):
                    old_value = list(graph.nodes(data = True))[i][1][self.attribute]
                    new_value = [old_value]
                    list(graph.nodes(data=True))[i][1][self.attribute] = new_value
            
            nx.set_edge_attributes(self.graph, "sliding window connection", "strategy")
        
                
        else:
            g =  self.strategy.to_graph(model.TimeseriesArrayStream(self.time_series))
            self.graph = g.graph

            for i in range(len(self.graph.nodes)):
                old_value = self.graph.nodes[i][self.attribute]
                new_value = [old_value]
                self.graph.nodes[i][self.attribute] = new_value
            

            hash = self.hash()
            mapping = {node: f"{hash}_{node}" for node in self.graph.nodes}
            self.graph = nx.relabel_nodes(self.graph, mapping)

            nx.set_edge_attributes(self.graph, strategy.strategy_name(), "strategy")


        return self
    
    def combine_identical_nodes(self):
        """Combines nodes that have same value of attribute self.attribute if graph is classical graph and
        nodes that are identical graphs if graph is created using sliding window mechanism."""
        self.orig_graph = self.graph.copy()
        if self.slid_win:

            for i, node_1 in enumerate(list(self.graph.nodes)):
                if node_1 not in self.graph:
                    continue

                for node_2 in list(self.graph.nodes)[i+1:]:
                    if node_2 == None:
                        break
                    if node_2 not in self.graph:
                        continue

                    if(set(list(node_1.edges)) == set(list(node_2.edges))):
                        self.graph = self.__combine_nodes_win(self.graph, node_1, node_2, self.attribute)
        else:

            for i, node_1 in enumerate(list(self.graph.nodes(data=True))):
                if node_1 not in self.graph:
                    continue

                for node_2 in list(self.graph.nodes(data=True))[i+1:]:
                    if node_2 == None:
                        break
                    if node_2 not in self.graph:
                        continue

                    if(node_1[self.attribute] == node_2[self.attribute]):
                        self.graph = self.__combine_nodes(self.graph, node_1, node_2, self.attribute)
            

        return self
    
    def __combine_nodes(self, graph, node_1, node_2, att):
        """Combines nodes node_1 and node_2."""
        node_1[att].append(node_2[att])
        for neighbor in list(graph.neighbors(node_2)):
            graph.add_edge(node_1, neighbor)
        
        graph.remove_node(node_2)
        return graph

    def __combine_nodes_win(self, graph, node_1, node_2, att):
        """Combines nodes node_1 and node_2, that are graphs."""
        for i in range(len(list(node_1.nodes(data=True)))):
            for j in range(len(list(node_2.nodes(data=True))[i][1][att])):
                list(node_1.nodes(data=True))[i][1][att].append(list(node_2.nodes(data=True))[i][1][att][j])
        
        for neighbor in list(graph.neighbors(node_2)):
            graph.add_edge(node_1, neighbor)

        graph.remove_node(node_2)
        return graph

    def draw(self, color = "black"):
        """Draws the created graph"""
        pos=nx.spring_layout(self. graph, seed=1)
        nx.draw(self.graph, pos, node_size=40, node_color=color)
        plt.show()
        return self
    
    def link(self, link_strategy):
        """Links nodes based on link_strategy. link_strategy is object of class Link."""
        self.graph = link_strategy.link(self.graph)
        return self
        
    def add_edge(self, node_1, node_2, weight=None):
        """Adds edge between node_1 and node_2."""
        if weight == None:
            self.graph.add_edge(list(self.graph.nodes)[node_1], list(self.graph.nodes)[node_2])
        else:
            self.graph.add_edge(list(self.graph.nodes)[node_1], list(self.graph.nodes)[node_2], weight = weight)
        return self
    
    def hash(self):
        """Returns unique hash of this graph."""
        str_to_hash = str(self.graph.nodes()) + str(self.graph.edges())
        return hashlib.md5(str_to_hash.encode()).hexdigest()




class GraphMaster:
    """Superclass of classes GraphSlidWin and Graph"""
    def __init__(self, graph, strategy):
        self.graph = graph
        self.next_node_strategy = "random"
        self.next_value_strategy = "random"
        self.skip_values = 0
        self.time_series_len = 100
        self.sequences = None
        self.walk = "one"
        self.switch_graphs = 1
        self.colors = None
        self.nodes = None
        self.data_nodes = None
        self.strategy = strategy
        self.att = 'value'
    
    def set_nodes(self, nodes, data_nodes):
        """Sets parameters to be used if we have multivariate graph and need nodes from specific original graph."""
        pass
    
    def set_attribute(self, att):
        self.att = att

    """Next 6 function set the parameters, that are later on used as a strategy for converting graph to time series."""
    """--->"""
    def walk_through_all(self):
        self.walk = "all"
        return self
    
    def change_graphs_every_x_steps(self, x):
        self.switch_graphs = x
        return self
    
    def choose_next_node(self, strategy):
        self.next_node_strategy = strategy
        return self
    
    def choose_next_value(self, strategy):
        self.next_value_strategy = strategy
        return self
    
    def skip_every_x_steps(self, x):
        self.skip_values = x
        return self
    
    def ts_length(self, x):
        self.time_series_len = x
        return self
    """<---"""

    def to_time_sequence(self):
        """Adjusts parameters nodes and data_nodes to fit function to_multiple_time_sequences."""
        pass
    
    def to_multiple_time_sequences(self):
        """Converts graph into time sequences."""
        pass

    def is_equal(self, graph_1, graph_2):
        """Compares two graphs if they are equal."""
        if(graph_1.nodes != graph_2.nodes): return False
        if(graph_1.edges != graph_2.edges): return False
        for i in range(len(graph_1.nodes)):
            if list(list(graph_1.nodes(data=True))[i][1][self.att]) != list(list(graph_2.nodes(data=True))[i][1][self.att]):
                    return False
        return True

    def plot_timeseries(self, sequence, title, x_legend, y_legend, color):
        """Function to sets parameters to draw time series."""
        plt.figure(figsize=(10, 6))
        plt.plot(sequence, linestyle='-', color=color)
        
        plt.title(title)
        plt.xlabel(x_legend)
        plt.ylabel(y_legend)
        plt.grid(True)

    def draw(self):
        """Draws time series."""
        if self.colors == None:
            self.colors = []
            for j in range(len(self.sequences)):
                self.colors.append("black")
        
        for j in range(len(self.sequences)):
            self.plot_timeseries(self.sequences[j], f"walk = {self.walk}, next_node_strategy = {self.next_node_strategy}, next_value = {self.next_value_strategy}", "Date", "Value", self.colors[j])
        plt.show()

class GraphSlidWin(GraphMaster):
    """Class that converts graphs made using sliding window mechanism back to time series"""
    def __init__(self, graph):
        super().__init__(graph, "slid_win")
    
    def set_nodes(self, nodes):
        self.nodes = nodes
        return self

    def to_time_sequence(self):
        self.nodes = [list(self.nodes)]
        return self.to_multiple_time_sequences()

    def to_multiple_time_sequences(self):
    
        self.sequences = [[] for _ in range(len(self.nodes))]

        current_nodes = [None for _ in range(len(self.nodes))]

        for i in range(len(self.nodes)):
            current_nodes[i] = self.nodes[i][0]
        
        dictionaries = [{} for _ in range(len(self.nodes))]
        for i in range(len(self.nodes)):
            for j in range(len(list(self.nodes[i]))):
                dictionaries[i][j] = 0

        
        strategy = None

        strategy = ChooseStrategySlidWin(self.walk, self.next_node_strategy, self.next_value_strategy, self.graph, self.nodes, dictionaries, self.att)

        i = 0
        while len(self.sequences[0]) < self.time_series_len:
            
            for j in range(len(self.sequences)):

                index = 0
                for i in range(len(list(self.nodes[j]))):
                    if(self.is_equal(current_nodes[j], list(self.graph.nodes)[i])):
                        index = i
                        break

                self.sequences[j] = strategy.append(self.sequences[j], current_nodes[j], j, index)
                if self.sequences[j][-1] == None:
                    return self
                

            for j in range(self.skip_values + 1):
                for k in range(len(current_nodes)):
                    
                    current_nodes[k] = strategy.next_node(i, k, current_nodes, self.switch_graphs)

                    if(current_nodes[k] == None):
                        return self
            
            """
            for k in range(len(current_nodes)):
                    current_nodes[k] = strategy.next_node(i, k, current_nodes, self.switch_graphs)
                    if(current_nodes[k] == None):
                        break
            """
            i += 1
        return self

class Graph(GraphMaster):
    """Class that turns ordinary graphs back to time series."""
    def __init__(self, graph):
        super().__init__(graph, "classic")
    
    def set_nodes(self, nodes, data_nodes):
        self.nodes = nodes
        self.data_nodes = data_nodes
        return self

    def to_time_sequence(self):
        self.nodes = [list(self.nodes)]
        self.data_nodes = [list(self.data_nodes)]
        return self.to_multiple_time_sequences()

    def to_multiple_time_sequences(self):
    
        self.sequences = [[] for _ in range(len(self.nodes))]

        current_nodes = [None for _ in range(len(self.nodes))]
        current_nodes_data = [None for _ in range(len(self.data_nodes))]

        for i in range(len(self.nodes)):
            current_nodes[i] = self.nodes[i][0]
            current_nodes_data[i] = self.data_nodes[i][0]
        
        dictionaries = [{} for _ in range(len(self.nodes))]
        for i in range(len(self.nodes)):
            for j in range(len(list(self.nodes[i]))):
                dictionaries[i][j] = 0

        strategy = ChooseStrategy(self.walk, self.next_node_strategy, self.next_value_strategy, self.graph, self.nodes, dictionaries, self.att)

        i = 0
        while len(self.sequences[0]) < self.time_series_len:
            
            for j in range(len(current_nodes)):

                index = 0

                for i in range(len(list(self.nodes[j]))):
                    if(current_nodes_data[j] == self.data_nodes[j][i]):
                        index = i
                        break
                
                self.sequences[j] = strategy.append(self.sequences[j], current_nodes_data[j], j, index)
                if self.sequences[j][-1] == None:
                    return

            for j in range(self.skip_values+1):
                for k in range(len(current_nodes)):
                    current_nodes[k] = strategy.next_node(i, k, current_nodes, self.switch_graphs)

                    new_index = self.nodes[k].index(current_nodes[k])
                    current_nodes_data[k] = self.data_nodes[k][new_index]
                    if(current_nodes[k] == None):
                        break
            
            i += 1
        return self

class ChooseStrategyMaster:
    def __init__(self, walk, next_node_strategy, value, graph, nodes, dictionaries, att):
        self.next_node_strategy = next_node_strategy
        self.walk = walk
        self.value = value
        self.graph = graph
        self.nodes = nodes
        self.dictionaries = dictionaries
        self.att = att
    
    def append_random(self, sequence, graph):
        """To a sequence appends a random value of a node it is currently on."""
        pass

    def append_lowInd(self, sequence, graph, graph_index, index):
        """To a sequence appends a successive value of a node it is currently on."""
        pass

    def append(self, sequence, graph, graph_index, index):
        if(self.value) == "random":
            return self.append_random(sequence, graph)
        elif self.value == "sequential" :
           return self.append_lowInd(sequence, graph, graph_index, index)
        else:
            print("you chose non-existent method of value selection")
            print("please choose between: random, sequential")
            return None
    
    def next_node_one_random(self, graph_index, node):
        """From neighbors of the previous node randomly chooses next node."""
        neighbors = set(self.graph.neighbors(node))
        neighbors = list(set(self.nodes[graph_index]) & neighbors)
        return random.choice(neighbors)

    def next_node_one_weighted(self, graph_index, node):
        """From neighbors of the previous node chooses next one based on number of connections between them."""
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
        """If we have multivariate graph this function walks on first ones and 
        for others graphs chooses next node based on neighbors of the node in first graph."""
        if self.next_node_strategy == "random":
            return self.next_node_one_random(graph_index, node)
        elif  self.next_node_strategy == "weighted":
            return self.next_node_one_weighted(graph_index, node)
        else:
            print("you chose non-existent next_node_strategy.")
            print("please choose between: random, weighted")
            return None
    
    def next_node_all_random(self, i, graph_index, nodes, switch):
        """From neighbors of the previous node randomly chooses next node."""
        index = int((i/switch) % len(nodes))
        neighbors = set(self.graph.neighbors(nodes[index]))
        
        neighbors = list(set(self.nodes[graph_index]) & neighbors)
        return random.choice(neighbors)
    
    def next_node_all_weighted(self, i, graph_index, nodes, switch):
        """From neighbors of the previous node chooses next one based on number of connections between them."""
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
        """If we have multivariate graph this function walks on all graphs and switches between them every 'switch' steps and
        for others graphs chooses next node based on neighbors of the node in cuurent graph."""
        if self.next_node_strategy == "random":
            return self.next_node_all_random(i, graph_index, nodes, switch)
        elif  self.next_node_strategy == "weighted":
            return self.next_node_all_weighted(i, graph_index, nodes, switch)
        else:
            print("you chose non-existent next_node_strategy.")
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
    
class ChooseStrategy(ChooseStrategyMaster):
    """Subclass that alters few methods to fit normal graph."""
    def __init__(self, walk, next_node_strategy, value, graph, nodes, dictionaries, att):
        super().__init__(walk, next_node_strategy, value, graph, nodes, dictionaries, att)
    
    def append_random(self, sequence, graph):
        index = random.randint(0, len(graph[1][self.att]) - 1)
        sequence.append(graph[1][self.att][index])
        return sequence
    
    def append_lowInd(self, sequence, graph, graph_index, index):
        if int(self.dictionaries[graph_index][index]/2) >= len(list(graph[1][self.att])):
            self.dictionaries[graph_index][index] = 0
        
        ind = int(self.dictionaries[graph_index][index]/2)
        sequence.append(graph[1][self.att][ind])
        self.dictionaries[graph_index][index] += 1
        return sequence

class ChooseStrategySlidWin(ChooseStrategyMaster):
    """Subclass that alters few methods to fit graph made using sliding window mechanism."""
    def __init__(self, walk, next_node_strategy, value, graph, nodes, dictionaries, att):
        super().__init__(walk, next_node_strategy, value, graph, nodes, dictionaries, att)
    
    def append_random(self, sequence, graph):

        nodes = list(graph.nodes(data = True))
        random.shuffle(nodes)

        for node in nodes:
            index = random.randint(0, len(node[1][self.att]) - 1)
            sequence.append(node[1][self.att][index])
        return sequence

    def append_lowInd(self, sequence, graph, graph_index, index):
        
        if int(self.dictionaries[graph_index][index]/2) >= len(list(list(graph.nodes(data=True))[0][1][self.att])):
            self.dictionaries[graph_index][index] = 0
    
        ind = int(self.dictionaries[graph_index][index]/2)

        for node in graph.nodes(data=True):
            sequence.append(node[1][self.att][ind])
    
        self.dictionaries[graph_index][index] += 1
        return sequence


class MultivariateTimeSeriesToGraph:
    """Class that combines multiple time series."""
    def __init__(self):
        self.graphs = {}
        self.multi_graph = None
        self.attribute = 'value'
    
    def set_attribute(self, att):
        self.attribute = att
        return self
    
    def link(self, link_strategy):
        """links nodes based on object link_strategy of class Link"""
        if self.multi_graph == None:
            self.multi_graph = link_strategy.link(self.graphs)
        else:
            self.multi_graph = link_strategy.link(self.multi_graph)
        return self
    
    def return_graph(self):
        """Returns graph."""
        return self.multi_graph

    def add(self, time_serie):
        """Adds object time_serie of class TimeSeriesToGraph to a dictionary."""
        self.graphs[time_serie.hash()] = time_serie.return_original_graph()
        return self
    
    def combine_identical_nodes_win(self):
        """Combines nodes that are identical graphs in a graph made using sliding window mechanism."""
        for graph in self.graphs.values():
            
            for i, node_1 in enumerate(list(graph.nodes)):
                if node_1 not in self.multi_graph:
                    continue

                for node_2 in list(graph.nodes)[i+1:]:
                    if node_2 == None:
                        break
                    if node_2 not in self.multi_graph:
                        continue

                    if(self.hash(node_1) == self.hash(node_2)):
                        graph = self.__combine_nodes_win(graph, node_1, node_2, self.attribute)
            
        
        return

    def hash(self, graph):
        """Returns unique hash of this graph."""
        str_to_hash = str(graph.nodes()) + str(graph.edges())
        return hashlib.md5(str_to_hash.encode()).hexdigest()

    def combine_identical_nodes(self):
        """Combines nodes that have same value of attribute self.attribute"""
        if isinstance(self.multi_graph, nx.MultiGraph):
            self.combine_identical_nodes_win()
            return self

        for graph in self.graphs.values():

            for i, node_1 in enumerate(list(graph.nodes(data=True))):
                if node_1 not in graph:
                    continue

                for node_2 in list(graph.nodes(data=True))[i+1:]:
                    if node_2 == None:
                        break
                    if node_2 not in graph:
                        continue

                    if(node_1[self.attribute] == node_2[self.attribute]):
                        graph = self.__combine_nodes(graph, node_1, node_2, self.attribute)
        return self
    
    def get_graph_nodes(self):
        """returns all nodes of graph"""
        nodes = []
        for graph in self.graphs.values():
            nodes.append(list(graph.nodes))
        
        return nodes

    def get_graph_nodes_data(self):
        """Returns all nodes of graphs with their data."""
        nodes = []
        for graph in self.graphs.values():
            nodes.append(list(graph.nodes(data = True)))
        
        return nodes
    
    def draw(self, color = "black"):
        """Draws graph."""
        pos=nx.spring_layout(self.multi_graph, seed=1)
        nx.draw(self.multi_graph, pos, node_size=40, node_color=color)

        plt.show()
        return self

    def __combine_nodes(self, graph, node_1, node_2, att):
        """Combines nodes node_1 and node_2."""
        node_1[att].append(node_2[att])
        for neighbor in list(graph.neighbors(node_2)):
            graph.add_edge(node_1, neighbor)
        
        graph.remove_node(node_2)
        return graph

    def __combine_nodes_win(self, graph, node_1, node_2, att):
        """Combines nodes node_1 and node_2, that are graphs."""
        
        for i in range(len(list(node_1.nodes(data=True)))):
            for j in range(len(list(node_2.nodes(data=True))[i][1][att])):
                list(node_1.nodes(data=True))[i][1][att].append(list(node_2.nodes(data=True))[i][1][att][j])
        
        for neighbor in list(self.multi_graph.neighbors(node_2)):
            self.multi_graph.add_edge(node_1, neighbor)
        
        #for neighbor in list(graph.neighbors(node_2)):
            #graph.add_edge(node_1, neighbor)

        self.multi_graph.remove_node(node_2)
        #graph.remove_node(node_2)
        return graph


#TODO: convert same_value into by_value(strategy), where strategy = same_value, same_quantile, proximity_criteria, etc.
#TODO: separate link into two objects depending on what type of graph you have.
