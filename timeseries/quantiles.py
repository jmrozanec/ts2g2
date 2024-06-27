from core.model import TimeseriesGraph
import networkx as nx
import itertools
import numpy as np


class TimeseriesToQuantileGraph:
    def __init__(self, Q):
        self.Q = Q

    def discretize_to_quantiles(self, time_series):
        quantiles = np.linspace(0, 1, self.Q)
        quantile_bins = np.quantile(time_series, quantiles)
        quantile_indices = np.digitize(time_series, quantile_bins) - 1  # Indices of quantiles for each observation
        return quantile_bins, quantile_indices

    def to_graph(self, time_series):
        quantile_bins, quantile_indices = self.discretize_to_quantiles(time_series)

        G = nx.DiGraph()

        for i in range(self.Q):
            G.add_node(i, label=f'Q{i + 1}')

        for (q1, q2) in zip(quantile_indices[:-1], quantile_indices[1:]):
            if G.has_edge(q1, q2):
                G[q1][q2]['weight'] += 1
            else:
                G.add_edge(q1, q2, weight=1)

        # Normalize the edge weights to represent transition probabilities
        for i in range(self.Q):
            total_weight = sum([G[i][j]['weight'] for j in G.successors(i)])
            if total_weight > 0:
                for j in G.successors(i):
                    G[i][j]['weight'] /= total_weight

        return G

class EdgeWeightingStrategy:
    def weight(self, G, x1, x2, y1, y2):
        return None


class EdgeWeightingStrategyNull(EdgeWeightingStrategy):
    def weight_edge(self, G, x1, x2, y1, y2):
        return None


class EdgeWeightingStrategyTransition():
    def weight_edge(self, G, x1, x2, y1, y2):
        return None


class TimeseriesToGraphStrategy:
    visibility_constraints = []
    graph_type = "undirected"
    edge_weighting_strategy = EdgeWeightingStrategyNull()

    def __init__(self, visibility_constraints, graph_type="undirected",
                 edge_weighting_strategy=EdgeWeightingStrategyNull()):
        self.visibility_constraints = visibility_constraints
        self.graph_type = graph_type
        self.edge_weighting_strategy = edge_weighting_strategy

    def to_graph(self, timeseries_stream):
        timeseries = timeseries_stream.read()

        G = nx.path_graph(len(timeseries), create_using=self.initialize_graph(self.graph_type))
        nx.set_node_attributes(G, dict(enumerate(timeseries)), "value")

        if any(vis_constr == "TimeseriesEdgeVisibilityConstraintsQuantile" for vis_constr in self.visibility_constraints):
            quantiles_structure = np.linspace(0, 1, self.visibility_constraints.q)
            quantile_bins = np.quantile(timeseries, quantiles_structure)
            quantile_indices = np.digitize(timeseries, quantile_bins) - 1
            timeseries = quantile_indices

        # Check all combinations of nodes n series
        for (x1, y1), (x2, y2) in itertools.combinations(enumerate(timeseries), 2):
            is_visible = True
            for visibility_constraint in self.visibility_constraints:
                is_obstructed = visibility_constraint.is_obstructed(timeseries, x1, x2, y1, y2)
                is_visible = is_visible and not is_obstructed

                if not is_visible:
                    break
            if is_visible:
                weight = self.edge_weighting_strategy.weight_edge(G, x1, x2, y1, y2)
                if weight is not None:
                    G.add_edge(x1, x2, weight=weight)
                else:
                    G.add_edge(x1, x2)

        return TimeseriesGraph(G)

    def initialize_graph(self, graph_type):
        if (graph_type == "undirected"):
            return nx.Graph()
        return nx.DiGraph()


class TimeseriesEdgeVisibilityConstraints:
    def is_obstructed(self, timeseries, x1, x2, y1, y2):
        return None


class EdgeAdditionStrategy:
    def add_edge(self, G, x1, x2, weight=None):
        return None


class TimeseriesEdgeVisibilityConstraintsNatural(TimeseriesEdgeVisibilityConstraints):
    limit = 0

    def __init__(self, limit=0):
        self.limit = limit

    def is_obstructed(self, timeseries, x1, x2, y1, y2):
        # check if any value between obstructs line of sight
        slope = (y2 - y1) / (x2 - x1)
        offset = y2 - slope * x2

        return any(
            y > slope * x + offset
            for x, y in enumerate(timeseries[x1 + self.limit + 1: x2], start=x1 + self.limit + 1)
        )


class TimeseriesEdgeVisibilityConstraintsQuantile(TimeseriesEdgeVisibilityConstraints):
    # timeseries = []
    q = 4

    def __init__(self, q=4):
        self.q = q
        # self.quantiles_structure = np.linspace(0, 1, self.q)
        # self.quantile_bins = np.quantile(timeseries, self.quantiles_structure)
        # self.quantile_indices = np.digitize(timeseries, self.quantile_bins) - 1

    def is_obstructed(self, quantile_indices, x1, x2, y1, y2):
        q1 = quantile_indices[x1]
        q2 = quantile_indices[x2]
        for i in range(x1 + 1, x2):
            qi = quantile_indices[i]
            if qi > min(q1, q2):
                return True
        return False
