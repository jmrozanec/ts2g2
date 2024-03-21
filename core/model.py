import copy

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