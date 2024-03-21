from core.model import TimeseriesArrayStream, Timeseries
from generation.strategies import RandomWalkWithRestartSequenceGenerationStrategy
from timeseries.strategies import TimeseriesToGraphStrategy, TimeseriesEdgeVisibilityConstraintsNatural, EdgeWeightingStrategyNull


def main():
    stream = TimeseriesArrayStream([2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3])
    timeseries = Timeseries(stream)
    ts2g = TimeseriesToGraphStrategy([TimeseriesEdgeVisibilityConstraintsNatural()], "undirected", EdgeWeightingStrategyNull())
    g = ts2g.to_graph(stream)
    sequence = g.to_sequence(RandomWalkWithRestartSequenceGenerationStrategy(), sequence_length=500)
    print('Original sequence: {}'.format(timeseries.to_sequence()))
    print('Generated sequence: {}'.format(sequence))

if __name__ == "__main__":
    main()