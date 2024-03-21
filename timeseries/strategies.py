"""
Time series graphs
"""

import math

from core.model import TimeseriesEdgeVisibilityConstraints,  EdgeAdditionStrategy, EdgeWeightingStrategy

class TimeseriesEdgeVisibilityConstraintsNatural(TimeseriesEdgeVisibilityConstraints):
    """
    Return a Natural Visibility Graph of a time series.

    A visibility graph converts a time series into a graph. The constructed graph
    uses integer nodes to indicate which event in the series the node represents.
    Edges are formed as follows: consider a bar plot of the series and view that
    as a side view of a landscape with a node at the top of each bar. An edge
    means that the nodes can be connected by a straight "line-of-sight" without
    being obscured by any bars between the nodes. The limit parameter introduces
    a limit of visibility and the nodes are connected only if, in addition to
    the natural visibility constraints, limit<x2-x1. Such a limit aims to reduce
    the effect of noise intrinsic to the data.

    The resulting graph inherits several properties of the series in its structure.
    Thereby, periodic series convert into regular graphs, random series convert
    into random graphs, and fractal series convert into scale-free networks [1]_.

    Parameters
    ----------
    limit : integer
       A limit established to the visibility between two bars of a time series.

    Returns
    -------
    Boolean
        Whether visibility is obstructed between the two bars of a time series.

    References
    ----------
    .. [1] Lacasa, Lucas, Bartolo Luque, Fernando Ballesteros, Jordi Luque, and Juan Carlos Nuno.
           "From time series to complex networks: The visibility graph." Proceedings of the
           National Academy of Sciences 105, no. 13 (2008): 4972-4975.
           https://www.pnas.org/doi/10.1073/pnas.0709247105
    .. [2] Zhou, Ting-Ting, Ningde Jin, Zhongke Gao and Yue-Bin Luo. “Limited penetrable visibility
           graph for establishing complex network from time series.” (2012).
           https://doi.org/10.7498/APS.61.030506
    """
    limit = 0
    def __init__(self, limit=0):
        self.limit = limit
    def is_obstructed(self, timeseries, x1, x2, y1, y2):
        # check if any value between obstructs line of sight
        slope = (y2 - y1) / (x2 - x1)
        offset = y2 - slope * x2

        return any(
            y >= slope * x + offset
            for x, y in enumerate(timeseries[x1 + self.limit + 1: x2], start=x1 + self.limit + 1)
        )

class TimeseriesEdgeVisibilityConstraintsHorizontal(TimeseriesEdgeVisibilityConstraints):
    """
    Return a Horizontal Visibility Graph of a time series.

    A visibility graph converts a time series into a graph. The constructed graph
    uses integer nodes to indicate which event in the series the node represents.
    Edges are formed as follows: consider a bar plot of the series and view that
    as a side view of a landscape with a node at the top of each bar. An edge
    means that the nodes can be connected by a horizontal "line-of-sight" without
    being obscured by any bars between the nodes [1]. The limit parameter introduces
    a limit of visibility and the nodes are connected only if, in addition to
    the natural visibility constraints, limit<x2-x1. Such a limit aims to reduce
    the effect of noise intrinsic to the data [2].

    Parameters
    ----------
    limit : integer
       A limit established to the visibility between two bars of a time series.

    Returns
    -------
    Boolean
        Whether visibility is obstructed between the two bars of a time series.

    References
    ----------
    .. [1] Luque, B., Lacasa, L., Ballesteros, F., & Luque, J. (2009).
           "Horizontal visibility graphs: Exact results for random time series."
           Physical Review E, 80(4), 046103.
           http://dx.doi.org/10.1103/PhysRevE.80.046103
    .. [2] Zhou, Ting-Ting, Ningde Jin, Zhongke Gao and Yue-Bin Luo. “Limited
           penetrable visibility graph for establishing complex network from
           time series.” (2012). https://doi.org/10.7498/APS.61.030506
    .. [3] Wang, M., Vilela, A.L.M., Du, R. et al. Exact results of the limited
           penetrable horizontal visibility graph associated to random time series
           and its application. Sci Rep 8, 5130 (2018).
           https://doi.org/10.1038/s41598-018-23388-1
    """
    limit = 0
    def __init__(self, limit=0):
        self.limit = limit
    def is_obstructed(self, timeseries, x1, x2, y1, y2):
        # check if any value between obstructs line of sight
        return any(
            y >= max(y1, y2)
            for x, y in enumerate(timeseries[x1 + self.limit + 1: x2], start=x1 + self.limit + 1)
        )

class TimeseriesEdgeVisibilityConstraintsVisibilityAngle(TimeseriesEdgeVisibilityConstraints):
    """
    Return a Parametric Visibility Graph of a time series.

    A visibility graph converts a time series into a graph. The constructed graph
    uses integer nodes to indicate which event in the series the node represents.
    Regardless the strategy used to define how edges are constructed, this class
    introduces additional constraints, ensuring that the angle between two
    observations meets a threshold parameter (visiblity_angle) [1]. Furthermore,
    additional constraints can be placed, ensuring that the absolute values of the
    angle between two observations and the threshold are considered [2].

    Parameters
    ----------
    visibility_angle : float
        A limit established to the visibility angle between two observations of a time series.
    consider_visibility_angle_absolute_value: bool, optional
        If True, the absolute values of the angle and the threshold are considered.

    Returns
    -------
    Boolean
        Whether visibility is obstructed between the two observations of a time series.

    References
    ----------
    .. [1] Bezsudnov, I. V., and A. A. Snarskii. "From the time series to the complex networks:
           The parametric natural visibility graph." Physica A: Statistical Mechanics and its
           Applications 414 (2014): 53-60.
           https://doi.org/10.1016/j.physa.2014.07.002
    .. [2] Supriya, S., Siuly, S., Wang, H., Cao, J., & Zhang, Y. (2016). Weighted visibility
           graph with complex network features in the detection of epilepsy. IEEE access, 4,
           6554-6566. https://doi.org/10.1109/ACCESS.2016.2612242
    """
    visibility_angle = 0
    consider_visibility_angle_absolute_value = True
    def __init__(self, visibility_angle=0, consider_visibility_angle_absolute_value=True):
        self.visibility_angle = visibility_angle
        self.consider_visibility_angle_absolute_value = consider_visibility_angle_absolute_value
    def is_obstructed(self, timeseries, x1, x2, y1, y2):
        slope = (y2 - y1) / (x2 - x1)

        angle = math.atan(slope)
        visibility_angle = self.visibility_angle
        if(self.consider_visibility_angle_absolute_value):
            angle = abs(angle)
            visibility_angle = abs(self.visibility_angle)

        return angle<visibility_angle

class EdgeAdditionStrategyUnweighted(EdgeAdditionStrategy):
    def add_edge(self, G, x1, x2, weight=None):
        G.add_edge(x1, x2)
class EdgeAdditionStrategyWeighted(EdgeAdditionStrategy):
    def add_edge(self, G, x1, x2, weight=None):
        G.add_edge(x1, x2, weight=weight)


class EdgeWeightingStrategyAngle(EdgeWeightingStrategy):
    absolute_value = True
    def __init__(self, absolute_value):
        self.absolute_value = absolute_value

    def weight_edge(self, G, x1, x2, y1, y2):
        slope = (y2 - y1) / (x2 - x1)
        angle = math.atan(slope)
        if self.absolute_value:
            return abs(angle)
        return angle

class EdgeWeightingStrategyNull(EdgeWeightingStrategy):
    def weight_edge(self, G, x1, x2, y1, y2):
        return None
