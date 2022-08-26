#!/usr/bin/env python3
# coding: utf-8
# Author:   Silas Dreier <silas.dreier@ifw-kiel.de>
"""Exercise Sheet 3"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.linear_model import LogisticRegression as Logit
from sklearn.linear_model import LinearRegression as OLS
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split, validation_curve
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix


# 1 Exercises for Supervise Machine Learning :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# (a)
df = pd.read_csv("./output/polynomials.csv", index_col=0)

# (b)
y = df["y"]
X = df. loc[:, df. columns != "y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# (c)
lm = OLS().fit(X_train, y_train)
ridge = Ridge(alpha=0.3).fit(X_train, y_train)
lasso = Lasso(alpha=0.3).fit(X_train, y_train)
lm.score(X_test, y_test)
ridge.score(X_test, y_test)
lasso.score(X_test, y_test)     # Yields the best predicition (R-Squared of 0.75)

# (d)
colnam = list(X.columns.values)
df = pd.DataFrame(lm.coef_, index=colnam, columns=["OLS"])
df["Lasso"] = lasso.coef_
df["Ridge"] = ridge.coef_
df.query("Lasso == 0 & Ridge != 0").shape[0]
# The Lasso coefficients are equal to zero in 449 cases where the ridge coefficients are not equal to zero

# (e)
# Plot Data
df1 = df
df1["OLS"] = lm.coef_.cumsum()
df1["Lasso"] = lasso.coef_.cumsum()
df1["Ridge"] = ridge.coef_.cumsum()

fig, ax = plt.subplots(figsize=(30, 20))
df1.plot.bar(ax=ax)
fig.savefig("./output/polynomials.pdf")



# 2  Neural Network Regression :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# (a)
diabetes = sklearn.datasets.load_diabetes()
y = diabetes["target"]
X = diabetes["data"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# (b)
# Scaling Data
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MLP Regression
algorithms = [("scaler", MinMaxScaler()),
            ("nn", MLPRegressor(solver="lbfgs", random_state=42, max_iter=1000, activation="identity"))]
pipe = Pipeline(algorithms, verbose = True)
para_grid = {"nn__hidden_layer_sizes": [(75, 75), (100, 100), (125, 125), (150, 150),
                                      (200, 200), (225, 225), (250, 250), (275, 275), (300, 300)],
            "nn__alpha": [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]}
grid = GridSearchCV(pipe, para_grid, cv=3, scoring='r2').fit(X_train, y_train)

# (c)
best_parameters = grid.best_params_
best_parameters["nn__alpha"] = [best_parameters["nn__alpha"]] # make 0.025 list-like
best_parameters_train_grid = GridSearchCV(pipe, best_parameters, cv=3, scoring='r2').fit(X_train, y_train)
print("On the training data, this model scores: {:.3f}".format(best_parameters_train_grid.best_score_))
best_parameters_test_grid = GridSearchCV(pipe, best_parameters, cv=3, scoring='r2').fit(X_test, y_test)
print("On the test data, this model scores: {:.3f}".format(best_parameters_test_grid.best_score_))
# Answer: Given that the model only scores 0.396 on the test data (compared to 0.467 on the training data)
# The generalisability of this (quite poor model to begin with) is low, given that the scores deviate by 7 percentage
# points.

# (d)
grid.best_estimator_
# Anything else: WTF?


#3. Neural Networks Classification::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#(a)
breast_cancer=load_breast_cancer()
y = breast_cancer["target"]
X = breast_cancer["data"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Min-Max Scaling
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classifying
algorithms = [("scaler", MinMaxScaler()),
                ("nn", MLPClassifier(activation='relu',max_iter=1000, random_state=42))]
pipe = Pipeline(algorithms, verbose = True)
para_grid = {"nn__hidden_layer_sizes": [(75, 75), (100, 100), (125, 125), (150, 150)],
                        "nn__alpha": [0.001, 0.01, 0.03, 0.015, 0.02]}
grid = GridSearchCV(pipe, para_grid, cv = 5,scoring='roc_auc').fit(X_train, y_train)

print("The Best model is:",grid.best_estimator_)
print("The Roc-Auc-Score for the best model is: {:.3f}".format(grid.best_score_))
# As the Roc-Auc-Score is at around almost 99%, the model should classify 99% of all cases correctly.
# To Check: Using model on test data
model=grid.best_estimator_
preds=model.predict(X_test)
print("The Roc-Auc-Score when used on test data is: {:.3f}".format(roc_auc_score(y_test, preds)),
        "\n The model therefore has a high generalizability, as it still predicts 97% of cancers cases correctly in out-of-sample data.")

# d)
matrix = confusion_matrix(y_test, preds)
ax = sns.heatmap(matrix, annot=True, xticklabels=breast_cancer["target_names"], yticklabels=breast_cancer["target_names"])
ax.figure.savefig("./output/nn_breast_confusion.pdf")