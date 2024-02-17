from core.model import TimeseriesArrayStream, Timeseries
from generation.strategies import RandomWalkGenerationStrategy
from timeseries.strategies import NaturalVisibilityGraphStrategy


def main():
    stream = TimeseriesArrayStream([2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3])
    timeseries = Timeseries(stream)
    g = timeseries.to_graph(NaturalVisibilityGraphStrategy())
    sequence = g.to_sequence(RandomWalkGenerationStrategy(), sequence_length=500)

if __name__ == "__main__":
    main()