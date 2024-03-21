import random

class GraphToTimeseriesStrategy:
    def to_sequence(self, graph, sequence_length):
        return None

class RandomNodeSequenceGenerationStrategy(GraphToTimeseriesStrategy):
    def to_sequence(self, graph, sequence_length):
        sequence = []
        nodes = [n for n in graph.nodes()]

        while len(sequence) < sequence_length:
            node = random.choice(nodes)
            sequence = sequence + [graph.nodes[node]['value']]

        return sequence

class RandomNodeNeighbourSequenceGenerationStrategy(GraphToTimeseriesStrategy):
    def to_sequence(self, graph, sequence_length):
        sequence = []
        nodes = [n for n in graph.nodes()]

        while len(sequence) < sequence_length:
            node = random.choice(nodes)
            neighbour = random.choice(graph.neighbors(node))
            sequence = sequence + [graph.nodes[neighbour]['value']]

        return sequence

class RandomDegreeNodeSequenceGenerationStrategy(GraphToTimeseriesStrategy):
    def to_sequence(self, graph, sequence_length):
        sequence = []
        nodesWeightedTuples = [(n, float(len(graph.neighbors(n)))/float(len(graph.nodes()))) for n in graph.nodes()]
        nodes = [n[0] for n in nodesWeightedTuples]
        nodeWeights = [n[1] for n in nodesWeightedTuples]

        while len(sequence) < sequence_length:
            node = random.choice(nodes, p=nodeWeights)
            sequence = sequence + [graph.nodes[node]['value']]

        return sequence

class RandomWalkSequenceGenerationStrategy(GraphToTimeseriesStrategy):
    def to_sequence(self, graph, sequence_length):
        sequence = []
        nodes = [n for n in graph.nodes()]
        first_node = random.choice(nodes)
        node = first_node
        while len(sequence) < sequence_length:
            sequence = sequence + [graph.nodes[node]['value']]
            nodes = [n for n in graph.neighbors(node)]
            if len(nodes) == 0:
                break
            else:
                node = random.choice(nodes)

        return sequence

class RandomWalkWithRestartSequenceGenerationStrategy(GraphToTimeseriesStrategy):
    def to_sequence(self, graph, sequence_length):
        sequence = []
        nodes = [n for n in graph.nodes()]
        first_node = random.choice(nodes)
        node = first_node
        while len(sequence) < sequence_length:
            sequence = sequence + [graph.nodes[node]['value']]
            if random.random() <0.15:
                node = first_node
            else:
                nodes = [n for n in graph.neighbors(node)]
                if len(nodes) == 0:
                    node = first_node
                else:
                    node = random.choice(nodes)

        return sequence
