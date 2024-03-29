# ts2g<sup>2</sup>

TS2G<sup>2</sup> stands for "timeseries to graphs and back". The library implements a variety of strategies to convert timeseries into graphs, and convert graphs into sequences.

    stream = TimeseriesArrayStream([2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3])
    timeseries = Timeseries(stream)
    g = timeseries.to_graph(NaturalVisibilityGraphStrategy())
    sequence = g.to_sequence(RandomWalkSequenceGenerationStrategy(), sequence_length=500)

Many of the methods implemented in this library, are described in _Silva, Vanessa Freitas, et al. "Time series analysis via network science: Concepts and algorithms." Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery 11.3 (2021): e1404._ Nevertheless, the library also includes additional techniques found in other works from the scientific literature.


### Timeseries to graph conversion

|   | Method                                                    | Reference                                                                                                                                                                                                                        | Implementing strategy                                      |
|---|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| 1 | Natural Visibility Graph                                  | [From time series to complex networks: The visibility graph.](https://www.pnas.org/doi/10.1073/pnas.0709247105)                                                                                                                  | NaturalVisibilityGraphStrategy                             |
| 2 | Horizontal Visibility Graph                               | [Horizontal visibility graphs: Exact results for random time series.](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.80.046103)                                                                                          | HorizontalVisibilityGraphStrategy                          |
| 3 | Directed Horizontal Visibility Graph                      | [Time series irreversibility: a visibility graph approach.](https://link.springer.com/article/10.1140/epjb/e2012-20809-8)                                                                                                        | DirectedHorizontalVisibilityGraphStrategy                  |
| 4 | Limited Penetrable Natural   Visibility Graph             | [Limited penetrable visibility graph for establishing complex network from time series](https://www.semanticscholar.org/paper/Limited-penetrable-visibility-graph-for-complex-Zhou-Jin/fe4a3d2f486021ee066f8a80472deef57d8aee71) | LimitedPenetrableNaturalVisibilityGraphStrategy            |
| 5 | Limited Penetrable Horizontal   Visibility Graph          | [Multiscale limited penetrable horizontal visibility graph for analyzing nonlinear time series](https://www.nature.com/articles/srep35622)                                                                                                                                                                                                                             | LimitedPenetrableHorizontalVisibilityGraphStrategy         |
| 6 | Limited Penetrable Directed   Horizontal Visibility Graph |                                                                                                                                                                                                                                  | LimitedPenetrableDirectedHorizontalVisibilityGraphStrategy |


 - Weighted Visibility Graphs
 - Parametric Natural/Horizontal Visibility Graphs


| 6 | Limited Penetrable Directed   Horizontal Visibility Graph |                                                                                                                                                                                                                                  | LimitedPenetrableDirectedHorizontalVisibilityGraphStrategy |
| 6 | Limited Penetrable Directed   Horizontal Visibility Graph |                                                                                                                                                                                                                                  | LimitedPenetrableDirectedHorizontalVisibilityGraphStrategy |
| 6 | Limited Penetrable Directed   Horizontal Visibility Graph |                                                                                                                                                                                                                                  | LimitedPenetrableDirectedHorizontalVisibilityGraphStrategy |
| 6 | Limited Penetrable Directed   Horizontal Visibility Graph |                                                                                                                                                                                                                                  | LimitedPenetrableDirectedHorizontalVisibilityGraphStrategy |
| 6 | Limited Penetrable Directed   Horizontal Visibility Graph |                                                                                                                                                                                                                                  | LimitedPenetrableDirectedHorizontalVisibilityGraphStrategy |
### Graphs to timeseries conversion

Graphs are converted back to timeseries by sampling node values from the graph following different strategies. The following strategies have been implemented so far:

 - random node
 - random node neighbour
 - random node degree 
 - random walk
 - random walk with restart
 - random walk with jump

