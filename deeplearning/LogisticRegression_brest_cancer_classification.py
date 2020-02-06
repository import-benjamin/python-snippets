#!/usr/bin/env python

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

cancer = load_breast_cancer()

print(f"cancer keys: {list(cancer.keys())}")
print(f"shape of cancer data : {cancer.data.shape}")
print(f"feature names : {cancer.feature_names}")

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

# C : regularization
logreg = LogisticRegression().fit(X_train, y_train)
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print(f"training score : {logreg.score(X_train, y_train)}")
print(f"test score : {logreg.score(X_test, y_test)}")

plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)

xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-5, 5)

plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")



plt.legend()

plt.show()
