import unittest
import numpy as np
import networkx as nx
from timeseries.ordinal_partition import TimeseriesToOrdinalPatternGraph

class TestTimeseriesToOrdinalPatternGraph(unittest.TestCase):
    def setUp(self):
        self.w = 3
        self.tau = 1
        self.obj = TimeseriesToOrdinalPatternGraph(self.w, self.tau)
        self.time_series = np.array([4, 2, 1, 3, 5])

    def test_embeddings(self):
        expected_result = np.array([
            [4, 2, 1],
            [2, 1, 3],
            [1, 3, 5]
        ])
        result = self.obj.embeddings(self.time_series)
        np.testing.assert_array_equal(result, expected_result)

    def test_ordinal_pattern(self):
        vector = np.array([4, 2, 1])
        expected_result = (2, 1, 0)
        result = self.obj.ordinal_pattern(vector)
        self.assertEqual(result, expected_result)

        vector = np.array([1, 3, 5])
        expected_result = (0, 1, 2)
        result = self.obj.ordinal_pattern(vector)
        self.assertEqual(result, expected_result)

    def test_to_graph(self):
        G = self.obj.to_graph(self.time_series, self.w, self.tau)
        expected_nodes = {
            (2, 1, 0),  # [4, 2, 1]
            (1, 0, 2),  # [2, 1, 3]
            (0, 1, 2)   # [1, 3, 5]
        }
        expected_edges = {
            ((2, 1, 0), (1, 0, 2)),
            ((1, 0, 2), (0, 1, 2))
        }

        self.assertEqual(set(G.nodes), expected_nodes)
        self.assertEqual(set(G.edges), expected_edges)

        # Check edge weights
        self.assertAlmostEqual(G[(2, 1, 0)][(1, 0, 2)]['weight'], 1/3)
        self.assertAlmostEqual(G[(1, 0, 2)][(0, 1, 2)]['weight'], 1/3)

if __name__ == '__main__':
    unittest.main()

