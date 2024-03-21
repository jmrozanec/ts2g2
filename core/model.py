import copy
from timeseries.strategies import EdgeWeightingStrategyNull, EdgeAdditionStrategyUnweighted
import itertools
import networkx as nx
import math

class TimeseriesStream:
  def read(self):
      return None

class Timeseries:
    def __init__(self, timeseries_stream):
        self.timeseries_stream = timeseries_stream

    def to_sequence(self):
        return self.timeseries_stream.read()

    def to_graph(self, timeseries_to_graph_strategy):
        return timeseries_to_graph_strategy.to_graph(self.timeseries_stream)

class TimeseriesGraph:
    def __init__(self, graph):
        self.graph = graph

    def to_sequence(self, graph_to_timeseries_strategy, sequence_length):
        return graph_to_timeseries_strategy.to_sequence(self.graph, sequence_length)

class TimeseriesToGraphStrategy:
    graph_type = "undirected"
    visibility_constraints = []
    edge_weighting_strategy = None
    edge_addition_strategy = None

    def __init__(self, graph_type, visibility_constraints, edge_weighting_strategy=EdgeWeightingStrategyNull(), edge_addition_strategy=EdgeAdditionStrategyUnweighted()):
        self.graph_type = graph_type
        self.visibility_constraints = visibility_constraints
        self.edge_weighting_strategy = edge_weighting_strategy
        self.edge_addition_strategy = edge_addition_strategy

    def to_graph(self, timeseries_stream):
        """
            Return a Visibility Graph encoding the particular time series.

            A visibility graph converts a time series into a graph. The constructed graph
            uses integer nodes to indicate which event in the series the node represents.
            Edges are formed as follows: consider a bar plot of the series and view that
            as a side view of a landscape with a node at the top of each bar. An edge
            means that the nodes can be connected somehow by a "line-of-sight" without
            being obscured by any bars between the nodes and respecting all the
            specified visibility constrains.

            Parameters
            ----------
            timeseries_stream : Sequence[Number]
                   A Time Series sequence (iterable and sliceable) of numeric values
                   representing values at regular points in time.

                Returns
                -------
                NetworkX Graph
                    The Natural Visibility Graph of the timeseries time series

                Examples
                --------
                >>> timeseries = [2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3]
                >>>
                ...     g = ts2g2.TimeseriesToGraphStrategy.to_graph(timeseries)
                ...     print(g)
                """
        timeseries = timeseries_stream.read()

        G = nx.path_graph(self.initialize_graph(self.graph_type), len(timeseries))
        nx.set_node_attributes(G, dict(enumerate(timeseries)), "value")

        is_visible = True
        # Check all combinations of nodes n series
        for (x1, y1), (x2, y2) in itertools.combinations(enumerate(timeseries), 2):
            for visibility_constraint in self.visibility_constraints:
                is_visible = is_visible and visibility_constraint.is_obstructed(timeseries, x1, x2, y1, y2)
                if not is_visible:
                    break
            if is_visible:
                weight = self.edge_weighting_strategy(timeseries)
                self.add_edge(G, x1, x2, weight)
        return TimeseriesGraph(G)

    def add_edge(self, G, x1, x2, weight):
        return self.edge_addition_strategy.add_edge(G, x1, x2, weight)

    def initialize_graph(self, graph_type):
        if(graph_type=="undirected"):
            return nx.Graph()
        return nx.DiGraph()


class GraphToTimeseriesStrategy:
    def to_sequence(self, graph, sequence_length):
        return None

class TimeseriesEdgeVisibilityConstraints:
    def is_obstructed(self, timeseries, x1, x2, y1, y2):
        return None

class EdgeAdditionStrategy:
    def add_edge(self, G, x1, x2, weight=None):
        return None

class EdgeWeightingStrategy:
    def weight(self, G, x1, x2, y1, y2):
        return None






class TimeseriesFileStream(TimeseriesStream):
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        with open(self.file_path, 'r') as file:
            data = file.read()
        return data

class TimeseriesArrayStream(TimeseriesStream):
    def __init__(self, array):
        self.array = copy.deepcopy(array)

    def read(self):
        return copy.deepcopy(self.array)