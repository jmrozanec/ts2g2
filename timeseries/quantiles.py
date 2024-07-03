import networkx as nx
import numpy as np

class TimeseriesToQuantileGraph:
    def __init__(self, Q):
            self.Q = Q

    def discretize_to_quantiles(self, time_series):
        quantiles = np.linspace(0, 1, self.Q + 1)
        quantile_bins = np.quantile(time_series, quantiles)
        quantile_bins[0] -= 1e-9
        quantile_indices = np.digitize(time_series, quantile_bins, right=True) - 1
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