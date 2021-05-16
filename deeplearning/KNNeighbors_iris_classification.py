import sys
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
import sklearn
import mglearn

# !! This script is not optimized.

print(f"Python version {sys.version}")
print(f"pandes version {pd.__version__}")
print(f"matplotlib version {matplotlib.__version__}")
print(f"numpy version {np.__version__}")
print(f"scipy version {sp.__version__}")
print(f"IPython version {IPython.__version__}")
print(f"scikit-learn version {sklearn.__version__}")
print(f"mglearn version {mglearn.__version__}")

from sklearn.datasets import load_iris

iris_dataset = load_iris()

print(f"iris_dataset keys : {iris_dataset.keys()}")
print(f"iris_dataset description : {iris_dataset['DESCR'][:225]}")
print(f"iris_dataset sample : \n{iris_dataset['data'][:5]}")

# train_test_split extract 75% of the dataset to train our model and keep the remaining 25% for test.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset["data"], iris_dataset["target"], random_state=0
)

print(f"X_train shape : {X_train.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"y_test shape : {y_test.shape}")

import matplotlib.pyplot as plt

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(
    iris_dataframe,
    c=y_train,
    figsize=(15, 15),
    marker="o",
    hist_kwds={"bins": 20},
    s=60,
    alpha=0.8,
    cmap=mglearn.cm3,
)
plt.show()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"Model accuracy (np mean) : {np.mean(y_pred == y_test)}")
print(f"Model accuracy (knn score) : {knn.score(X_test, y_test)}")
