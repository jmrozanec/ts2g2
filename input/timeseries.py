"""
Time series graphs
"""

import itertools
import networkx as nx

def natural_visibility_graph(timeseries):
    """
    Return a Natural Visibility Graph of an input Time Series.

    A visibility graph converts a time series into a graph. The constructed graph
    uses integer nodes to indicate which event in the series the node represents.
    Edges are formed as follows: consider a bar plot of the series and view that
    as a side view of a landscape with a node at the top of each bar. An edge
    means that the nodes can be connected by a straight "line-of-sight" without
    being obscured by any bars between the nodes.

    The resulting graph inherits several properties of the series in its structure.
    Thereby, periodic series convert into regular graphs, random series convert
    into random graphs, and fractal series convert into scale-free networks [1]_.

    Parameters
    ----------
    series : Sequence[Number]
       A Time Series sequence (iterable and sliceable) of numeric values
       representing values at regular points in time.

    Returns
    -------
    NetworkX Graph
        The Natural Visibility Graph of the input time series

    Examples
    --------
    >>> timeseries = [2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3]
    >>>
    ...     g = tsg.natural_visibility_graph(timeseries)
    ...     print(g)
    Graph with 12 nodes and 18 edges

    References
    ----------
    .. [1] Lacasa, Lucas, Bartolo Luque, Fernando Ballesteros, Jordi Luque, and Juan Carlos Nuno.
           "From time series to complex networks: The visibility graph." Proceedings of the
           National Academy of Sciences 105, no. 13 (2008): 4972-4975.
           https://www.pnas.org/doi/10.1073/pnas.0709247105
    """
    return nx.visibility_graph(timeseries)


def horizontal_visibility_graph(timeseries):
    """
    Return a Horizontal Visibility Graph of an input Time Series.

    The Horizontal Visibility Graph converts a time series into a graph.
    The constructed graph uses integer nodes to indicate which event in the series the node represents.
    The edges are formed as follows: consider a bar plot of the series and view that
    as a side view of a landscape with a node at the top of each bar. An edge
    means that the nodes can be connected by a horizontal "line-of-sight" without
    being obscured by any bars between the nodes.

    The resulting graph inherits several properties of the series in its structure.
    Thereby, periodic series convert into regular graphs, random series convert
    into random graphs, and fractal series convert into scale-free networks [1]_.

    Parameters
    ----------
    series : Sequence[Number]
       A Time Series sequence (iterable and sliceable) of numeric values
       representing values at regular points in time.

    Returns
    -------
    NetworkX Graph
        The Horizontal Visibility Graph of the input time series

    Examples
    --------
    >>> timeseries = [2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3]
    >>>
    ...     g = tsg.horizontal_visibility_graph(timeseries)
    ...     print(g)
    Graph with 12 nodes and 18 edges

    References
    ----------
    .. [1] Luque, B., Lacasa, L., Ballesteros, F., & Luque, J. (2009).
           "Horizontal visibility graphs: Exact results for random time series."
           Physical Review E, 80(4), 046103.
    """
    # Sequential values are always connected
    G = nx.path_graph(len(timeseries))
    nx.set_node_attributes(G, dict(enumerate(timeseries)), "value")

    # Check all combinations of nodes n series
    for (n1, t1), (n2, t2) in itertools.combinations(enumerate(timeseries), 2):
        obstructed = any(
            t >= max(t1, t2)
            for n, t in enumerate(timeseries[n1 + 1 : n2], start=n1 + 1)
        )

        if not obstructed:
            G.add_edge(n1, n2)

    return G


def directed_horizontal_visibility_graph(timeseries):
    """
    Return a Directed Horizontal Visibility Graph of an input Time Series.

    The Horizontal Visibility Graph converts a time series into a graph.
    The constructed graph uses integer nodes to indicate which event in the series the node represents.
    The edges are formed as follows: consider a bar plot of the series and view that
    as a side view of a landscape with a node at the top of each bar. An edge
    means that the nodes can be connected by a horizontal "line-of-sight" without
    being obscured by any bars between the nodes.

    The resulting graph inherits several properties of the series in its structure.
    Thereby, periodic series convert into regular graphs, random series convert
    into random graphs, and fractal series convert into scale-free networks [1]_.

    Parameters
    ----------
    series : Sequence[Number]
       A Time Series sequence (iterable and sliceable) of numeric values
       representing values at regular points in time.

    Returns
    -------
    NetworkX Graph
        The Directed Horizontal Visibility Graph of the input time series

    Examples
    --------
    >>> timeseries = [2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3]
    >>>
    ...     g = tsg.directed_horizontal_visibility_graph(timeseries)
    ...     print(g)
    Graph with 12 nodes and 18 edges

    References
    ----------
    .. [1] Lacasa, L., Nunez, A., Roldan, ´ E., Parrondo, J. M., and Luque, B. (2012).
           "Time series irreversibility: a visibility graph approach."
           The European Physical Journal B, 85(6):217.
    """
    # Sequential values are always connected
    G = nx.path_graph(len(timeseries), create_using=nx.DiGraph())
    nx.set_node_attributes(G, dict(enumerate(timeseries)), "value")

    # Check all combinations of nodes n series
    for (n1, t1), (n2, t2) in itertools.combinations(enumerate(timeseries), 2):
        obstructed = any(
            t >= max(t1, t2)
            for n, t in enumerate(timeseries[n1 + 1 : n2], start=n1 + 1)
        )

        if not obstructed:
            G.add_edge(n1, n2)

    return G

=================

def limited_penetrable_natural_visibility_graph(timeseries, l=1):
    """
    Return a Natural Visibility Graph of an input Time Series.

    A visibility graph converts a time series into a graph. The constructed graph
    uses integer nodes to indicate which event in the series the node represents.
    Edges are formed as follows: consider a bar plot of the series and view that
    as a side view of a landscape with a node at the top of each bar. An edge
    means that the nodes can be connected by a straight "line-of-sight" without
    being obscured by any bars between the nodes.

    The resulting graph inherits several properties of the series in its structure.
    Thereby, periodic series convert into regular graphs, random series convert
    into random graphs, and fractal series convert into scale-free networks [1]_.

    Parameters
    ----------
    series : Sequence[Number]
       A Time Series sequence (iterable and sliceable) of numeric values
       representing values at regular points in time.

    Returns
    -------
    NetworkX Graph
        The Natural Visibility Graph of the input time series

    Examples
    --------
    >>> timeseries = [2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3]
    >>>
    ...     g = tsg.natural_visibility_graph(timeseries)
    ...     print(g)
    Graph with 12 nodes and 18 edges

    References
    ----------
    .. [1] Lacasa, Lucas, Bartolo Luque, Fernando Ballesteros, Jordi Luque, and Juan Carlos Nuno.
           "From time series to complex networks: The visibility graph." Proceedings of the
           National Academy of Sciences 105, no. 13 (2008): 4972-4975.
           https://www.pnas.org/doi/10.1073/pnas.0709247105
    """
    return nx.visibility_graph(timeseries)


def limited_penetrable_horizontal_visibility_graph(timeseries, l=1):
    """
    Return a Horizontal Visibility Graph of an input Time Series.

    The Horizontal Visibility Graph converts a time series into a graph.
    The constructed graph uses integer nodes to indicate which event in the series the node represents.
    The edges are formed as follows: consider a bar plot of the series and view that
    as a side view of a landscape with a node at the top of each bar. An edge
    means that the nodes can be connected by a horizontal "line-of-sight" without
    being obscured by any bars between the nodes.

    The resulting graph inherits several properties of the series in its structure.
    Thereby, periodic series convert into regular graphs, random series convert
    into random graphs, and fractal series convert into scale-free networks [1]_.

    Parameters
    ----------
    series : Sequence[Number]
       A Time Series sequence (iterable and sliceable) of numeric values
       representing values at regular points in time.

    Returns
    -------
    NetworkX Graph
        The Horizontal Visibility Graph of the input time series

    Examples
    --------
    >>> timeseries = [2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3]
    >>>
    ...     g = tsg.horizontal_visibility_graph(timeseries)
    ...     print(g)
    Graph with 12 nodes and 18 edges

    References
    ----------
    .. [1] Luque, B., Lacasa, L., Ballesteros, F., & Luque, J. (2009).
           "Horizontal visibility graphs: Exact results for random time series."
           Physical Review E, 80(4), 046103.
    """
    # Sequential values are always connected
    G = nx.path_graph(len(timeseries))
    nx.set_node_attributes(G, dict(enumerate(timeseries)), "value")

    # Check all combinations of nodes n series
    for (n1, t1), (n2, t2) in itertools.combinations(enumerate(timeseries), 2):
        obstructed = any(
            (t >= max(t1, t2)) and (n-n1 <= l)
            for n, t in enumerate(timeseries[n1 + 1 : n2], start=n1 + 1)
        )

        if not obstructed:
            G.add_edge(n1, n2)

    return G


def limited_penetrable_directed_horizontal_visibility_graph(timeseries, l=1):
    """
    Return a Directed Horizontal Visibility Graph of an input Time Series.

    The Horizontal Visibility Graph converts a time series into a graph.
    The constructed graph uses integer nodes to indicate which event in the series the node represents.
    The edges are formed as follows: consider a bar plot of the series and view that
    as a side view of a landscape with a node at the top of each bar. An edge
    means that the nodes can be connected by a horizontal "line-of-sight" without
    being obscured by any bars between the nodes.

    The resulting graph inherits several properties of the series in its structure.
    Thereby, periodic series convert into regular graphs, random series convert
    into random graphs, and fractal series convert into scale-free networks [1]_.

    Parameters
    ----------
    series : Sequence[Number]
       A Time Series sequence (iterable and sliceable) of numeric values
       representing values at regular points in time.

    Returns
    -------
    NetworkX Graph
        The Directed Horizontal Visibility Graph of the input time series

    Examples
    --------
    >>> timeseries = [2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3]
    >>>
    ...     g = tsg.directed_horizontal_visibility_graph(timeseries)
    ...     print(g)
    Graph with 12 nodes and 18 edges

    References
    ----------
    .. [1] Lacasa, L., Nunez, A., Roldan, ´ E., Parrondo, J. M., and Luque, B. (2012).
           "Time series irreversibility: a visibility graph approach."
           The European Physical Journal B, 85(6):217.
    """
    # Sequential values are always connected
    G = nx.path_graph(len(timeseries), create_using=nx.DiGraph())
    nx.set_node_attributes(G, dict(enumerate(timeseries)), "value")

    # Check all combinations of nodes n series
    for (n1, t1), (n2, t2) in itertools.combinations(enumerate(timeseries), 2):
        obstructed = any(
            (t >= max(t1, t2)) and (n-n1 <= l)
            for n, t in enumerate(timeseries[n1 + 1 : n2], start=n1 + 1)
        )

        if not obstructed:
            G.add_edge(n1, n2)

    return G
