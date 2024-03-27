import unittest
from unittest.mock import Mock
from core.model import TimeseriesGraph
from timeseries.strategies import TimeseriesToGraphStrategy

class TestToGraphMethod(unittest.TestCase):

    def test_to_graph(self):
        class MockVisibilityConstraint:
            def is_obstructed(self, timeseries, x1, x2, y1, y2):
                return False  # Mock always returns False for visibility

        # Set up the test
        self.strategy_instance = TimeseriesToGraphStrategy([MockVisibilityConstraint()])

        # Call the to_graph method with the mock timeseries stream
        result_graph = self.strategy_instance.to_graph([1, 2, 3, 4, 5])

        # Check if the result is an instance of TimeseriesGraph
        self.assertIsInstance(result_graph, TimeseriesGraph)

        # Check if the graph has been constructed correctly
        expected_nodes = [0, 1, 2, 3, 4]
        expected_edges = [(0, 1, {'weight': 1}), (1, 2, {'weight': 1}),
                          (2, 3, {'weight': 1}), (3, 4, {'weight': 1})]
        self.assertEqual(sorted(result_graph.graph.nodes()), expected_nodes)
        self.assertEqual(sorted(result_graph.graph.edges(data=True)), expected_edges)

if __name__ == '__main__':
    unittest.main()
