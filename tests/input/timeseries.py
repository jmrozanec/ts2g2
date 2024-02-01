import unittest
import networkx as nx
from input import timeseries

class TimeseriesTest(unittest.TestCase):
    def test_natural_visibility_graph(self):
        # G = nx.
        self.assertEqual(timeseries.natural_visibility_graph([1, 2, 3]), 4)
