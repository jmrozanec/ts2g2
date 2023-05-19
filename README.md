# time-series-generator
A scalable time series generator.

    vertexes = given_file().with_timestamps_column().with_values_column().generate_graph()
    ts = given_graph(vertexes).with_start_value(random/someval).generate_time_series(how_many, how_long)
    #ts = given_graph(vertexes).with_start_value(random/someval).with_black_swans(n, length).generate_time_series(how_many, how_long)

## TODO
 - [X] create graph from time series
 - [X] generate time series from graph
 - [ ] create library of possible graphs/time-series
 - [ ] create a DSL, so that we can read a ts, leverage existing library, consider black swan events, etc.
