import random

from core.model import GraphToTimeseriesStrategy


class RandomWalkGenerationStrategy(GraphToTimeseriesStrategy):
    def to_sequence(self, graph, sequence_length):
        sequence = []
        nodes = [n for n in graph.nodes()]

        while len(sequence) < sequence_length:
            node = random.choice(nodes)
            sequence = sequence + [graph.nodes[node]['value']]
            nodes = [n for n in graph.neighbors(node)]

        return sequence
