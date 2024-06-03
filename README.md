# ts2g<sup>2</sup>

TS2G<sup>2</sup> stands for "timeseries to graphs and back". The library implements a variety of strategies to convert timeseries into graphs, and convert graphs into sequences.

    stream = TimeseriesArrayStream([2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3])
    timeseries = Timeseries(stream)
    g = timeseries.to_graph(NaturalVisibilityGraphStrategy())
    sequence = g.to_sequence(RandomWalkSequenceGenerationStrategy(), sequence_length=500)

Many of the methods implemented in this library, are described in _Silva, Vanessa Freitas, et al. "Time series analysis via network science: Concepts and algorithms." Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery 11.3 (2021): e1404._ Nevertheless, the library also includes additional techniques found in other works from the scientific literature.

This package is being developed as part of the [Graph-Massivizer](https://graph-massivizer.eu/) project. 
The package is a joint effort between the [Jožef Stefan Institute](https://www.ijs.si/), the [University of Twente](https://www.utwente.nl/en/), the [Vrije Universiteit Amsterdam](https://vu.nl/en), the [University of Klagenfurt](https://www.aau.at/en/), the [University of Bologna](https://www.unibo.it/en), and [Peracton](https://peracton.com/).


### Timeseries to graph conversion

#### Implemented features

<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt" rowspan="3">#</th>
    <th class="tg-7btt" rowspan="3">Visibility Graph</th>
    <th class="tg-7btt" colspan="3">Graph type</th>
    <th class="tg-7btt" colspan="4">Constraints</th>
  </tr>
  <tr>
    <th class="tg-7btt" rowspan="2">Undirected</th>
    <th class="tg-7btt" rowspan="2">Directed</th>
    <th class="tg-7btt" rowspan="2">Weighted</th>
  </tr>
  <tr>
    <th class="tg-7btt">Penetration</th>
    <th class="tg-7btt">Angle</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7btt">1</td>
    <td class="tg-0pky">Natural Visibility Graph</td>
    <td class="tg-0pky">
        X
    </td>
    <td class="tg-0pky">
      <!-- directed -->
      X
    </td>
    <td class="tg-0pky">
      <!-- weighted -->
      X
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: penetration -->
      X
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: angle -->
      X
    </td>
  </tr>
  <tr>
    <td class="tg-7btt">2</td>
    <td class="tg-0pky">Horizontal Visibility Graph</td>
    <td class="tg-0pky">
      X
    </td>
    <td class="tg-0pky">
        <!-- directed -->
        X
    </td>
    <td class="tg-0pky">
      <!-- weighted -->
      X 
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: penetration -->
      X
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: angle -->
      X
    </td>
  </tr>
  <tr>
    <td class="tg-7btt">3</td>
    <td class="tg-0pky">Difference Visibility Graph</td>
    <td class="tg-0pky">
        <!-- undirected -->
    </td>
    <td class="tg-0pky">
      <!-- directed -->
    </td>
    <td class="tg-0pky">
      <!-- weighted -->
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: penetration -->
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: angle -->
    </td>
  </tr>
</tbody>
</table>


#### References table

<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt" rowspan="3">#</th>
    <th class="tg-7btt" rowspan="3">Visibility Graph</th>
    <th class="tg-7btt" colspan="3">Graph type</th>
    <th class="tg-7btt" colspan="4">Constraints</th>
  </tr>
  <tr>
    <th class="tg-7btt" rowspan="2">Undirected</th>
    <th class="tg-7btt" rowspan="2">Directed</th>
    <th class="tg-7btt" rowspan="2">Weighted</th>
  </tr>
  <tr>
    <th class="tg-7btt">Penetration</th>
    <th class="tg-7btt">Angle</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7btt">1</td>
    <td class="tg-0pky">Natural Visibility Graph</td>
    <td class="tg-0pky">
        <a href="https://www.pnas.org/doi/10.1073/pnas.0709247105">ref</a>
    </td>
    <td class="tg-0pky">
      <!-- directed -->
      <a href="https://link.springer.com/article/10.1140/epjb/e2012-20809-8">ref</a>
    </td>
    <td class="tg-0pky">
      <!-- weighted -->
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: penetration -->
      <a href="https://www.semanticscholar.org/paper/Limited-penetrable-visibility-graph-for-complex-Zhou-Jin/fe4a3d2f486021ee066f8a80472deef57d8aee71">ref</a>
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: angle -->
      <a href="https://doi.org/10.1109/ACCESS.2016.2612242">ref</a>, 
      <a href="https://doi.org/10.1016/j.physa.2014.07.002">ref</a>
    </td>
  </tr>
  <tr>
    <td class="tg-7btt">2</td>
    <td class="tg-0pky">Horizontal Visibility Graph</td>
    <td class="tg-0pky">
      <a href="https://journals.aps.org/pre/abstract/10.1103/PhysRevE.80.046103">ref</a>
    </td>
    <td class="tg-0pky">
      <a href="https://link.springer.com/article/10.1140/epjb/e2012-20809-8">ref</a>
      <!-- directed -->
    </td>
    <td class="tg-0pky">
      <!-- weighted -->
      <a href="https://doi.org/10.1109/ACCESS.2016.2612242">ref</a> 
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: penetration -->
      <a href="https://www.semanticscholar.org/paper/Limited-penetrable-visibility-graph-for-complex-Zhou-Jin/fe4a3d2f486021ee066f8a80472deef57d8aee71">ref</a>, 
      <a href="https://www.nature.com/articles/srep35622">ref</a>
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: angle -->
    </td>
  </tr>
  <tr>
    <td class="tg-7btt">3</td>
    <td class="tg-0pky">Difference Visibility Graph</td>
    <td class="tg-0pky">
        <!-- undirected -->
    </td>
    <td class="tg-0pky">
      <!-- directed -->
    </td>
    <td class="tg-0pky">
      <!-- weighted -->
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: penetration -->
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: angle -->
    </td>
  </tr>
</tbody>
</table>



### Graphs to timeseries conversion

Graphs are converted back to timeseries by sampling node values from the graph following different strategies. The following strategies have been implemented so far:

 - random node
 - random node neighbour
 - random node degree 
 - random walk
 - random walk with restart
 - random walk with jump


## Publications

When using this work for research purposes, we would appreciate it if the following references could be included:


Below we provide a curated list of papers related to our research in this area:

