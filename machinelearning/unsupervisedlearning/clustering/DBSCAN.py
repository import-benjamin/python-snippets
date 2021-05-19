"""Compare clustering algorithm provided by scikit-learn.

Requirements
------------

I'm using GTK3Cairo has matplotlib backend on fedora 34. This might require to install cairocffi:

$ pip3 install cairocffi pycairo

See also
--------

https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf


"""

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from pandas import DataFrame

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# The DBSCAN algorithm views clusters as areas of high density separated by areas of low density. Due to this rather generic view, clusters found by DBSCAN can be any shape, as opposed to k-means which assumes that clusters are convex shaped. The central component to the DBSCAN is the concept of core samples, which are samples that are in areas of high density. A cluster is therefore a set of core samples, each close to each other (measured by some distance measure) and a set of non-core samples that are close to a core sample (but are not themselves core samples). There are two parameters to the algorithm, min_samples and eps, which define formally what we mean when we say dense. Higher min_samples or lower eps indicate higher density necessary to form a cluster.
from sklearn.cluster import DBSCAN

# The OPTICS algorithm shares many similarities with the DBSCAN algorithm, and can be considered a generalization of DBSCAN that relaxes the eps requirement from a single value to a value range. The key difference between DBSCAN and OPTICS is that the OPTICS algorithm builds a reachability graph, which assigns each sample both a reachability_ distance, and a spot within the cluster ordering_ attribute; these two attributes are assigned when the model is fitted, and are used to determine cluster membership.
from sklearn.cluster import OPTICS

# The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares (see below). This algorithm requires the number of clusters to be specified. It scales well to large number of samples and has been used across a large range of application areas in many different fields.
from sklearn.cluster import KMeans

# The Birch builds a tree called the Clustering Feature Tree (CFT) for the given data. The data is essentially lossy compressed to a set of Clustering Feature nodes (CF Nodes). The CF Nodes have a number of subclusters called Clustering Feature subclusters (CF Subclusters) and these CF Subclusters located in the non-terminal CF Nodes can have CF Nodes as children.
from sklearn.cluster import Birch

# In practice Spectral Clustering is very useful when the structure of the individual clusters is highly non-convex, or more generally when a measure of the center and spread of the cluster is not a suitable description of the complete cluster, such as when clusters are nested circles on the 2D plane.
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles

from sklearn.metrics import rand_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import completeness_score
from sklearn.metrics import mutual_info_score

from rich.table import Table
from rich.console import Console
import logging
from rich.logging import RichHandler

matplotlib.use("GTK3Cairo")

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger(__name__)

console = Console()

datasets = [
    (make_blobs, {"n_samples": 400, "centers": 2}),
    (make_moons, {"n_samples": 400}),
    (make_circles, {"n_samples": 400}),
]

algorithms = [
    (DBSCAN, {"eps": 0.3, "min_samples": 10, "n_jobs": -1}),
    (OPTICS, {"n_jobs": -1}),
    (KMeans, {"n_clusters": 2}),
    (Birch, {"n_clusters": 2}),
    (SpectralClustering, {"n_clusters": 2}),
]

metrics = [
    rand_score,
    fowlkes_mallows_score,
    adjusted_mutual_info_score,
    completeness_score,
    mutual_info_score,
]

datasets_len = len(datasets)
algorithms_len = len(algorithms)

tables_resume = []

with console.status("[bold green]Fit model...") as status:

    fig, axes = plt.subplots(datasets_len, algorithms_len + 1, figsize=(12, 12))
    fig.suptitle("Clustering comparison")

    standard_scaler = PCA()

    for d_index, (d_func, d_args) in enumerate(datasets):
        log.info(f"{d_index+1}/{datasets_len} Generaring {d_func.__name__} dataset")

        dataset_resume = Table("Algorithm", title=d_func.__name__.title())
        for metric_func in metrics:
            dataset_resume.add_column(metric_func.__name__.title())

        data, labels = d_func(**d_args)
        dataset = standard_scaler.fit_transform(data)
        dataframe = DataFrame(dataset, columns=["x", "y"])

        # axes[d_index][-1].scatter(data[:,0], data[:,0], label = "initial")
        plot = axes[d_index][-1].scatter(
            dataframe["x"], dataframe["y"], c=labels, cmap="tab20c"
        )
        axes[d_index][-1].set_title("dataset")
        axes[d_index][-1].legend(*plot.legend_elements(), fancybox=True)

        log.info(f"Start comparing clustering algorithmes")

        for a_index, (algorithm, arguments) in enumerate(algorithms):
            log.info(
                f"{d_index+1}/{datasets_len} - {a_index+1}/{algorithms_len} {algorithm.__name__.title()}"
            )
            clustering = algorithm(**arguments)
            label_preds = clustering.fit_predict(dataset)

            a_plot = axes[d_index][a_index].scatter(
                dataframe["x"], dataframe["y"], c=label_preds, cmap="tab20c"
            )
            axes[d_index][a_index].legend(*a_plot.legend_elements())

            metrics_result = [str(func(labels, label_preds)) for func in metrics]
            dataset_resume.add_row(algorithm.__name__, *metrics_result)

            axes[d_index][a_index].set_title(f"{algorithm.__name__}")

        tables_resume.append(dataset_resume)

console.rule("resume")
for t in tables_resume:
    console.print(t)

plt.show()
