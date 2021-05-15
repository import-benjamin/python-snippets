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
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn.metrics import rand_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import completeness_score
from sklearn.metrics import mutual_info_score

from rich.console import Console
import logging
from rich.logging import RichHandler

matplotlib.use('GTK3Cairo')

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger(__name__)

console = Console()

dataset, labels = make_blobs(
    n_samples=[233, 246, 342],
)

scaler = StandardScaler()
dataset = scaler.fit_transform(dataset)

dataframe = DataFrame(dataset, columns=["x", "y"])

# console.print(dataframe)

algorithms = [
    (DBSCAN, {"eps": 0.3, "min_samples": 10, "n_jobs": -1}),
    (OPTICS, {"eps": 0.3, "min_samples": 10, "n_jobs": -1}),
    (KMeans, {"n_clusters": 3})
]

with console.status("[bold green]Fit model...") as status:
    
    log.info(f"Start comparing clustering algorithmes")

    fig, axes = plt.subplots(1, len(algorithms)+1)

    sscatter = axes[-1].scatter(dataframe["x"], dataframe["y"], c=labels, cmap="tab20c")
    axes[-1].set_title("True values")
    axes[-1].legend(*sscatter.legend_elements())
    
    for index, (algorithm, arguments) in enumerate(algorithms):
        clustering = algorithm(**arguments)
        label_preds = clustering.fit_predict(dataset)

        scatter = axes[index].scatter(dataframe["x"], dataframe["y"], c=label_preds, cmap="tab20c")
        axes[index].legend(*scatter.legend_elements())
        
        log.info(f"Metrics for {algorithm.__name__}")

        for func in [ rand_score, fowlkes_mallows_score, adjusted_mutual_info_score, completeness_score, mutual_info_score ]:
            log.info(f"metric {func.__name__} scored: {func(labels, label_preds)}")
        
        log.info(f"{algorithm.__name__} is done")

        axes[index].set_title(f"{algorithm.__name__}")

    fig.suptitle("Clustering comparison")
plt.show()

