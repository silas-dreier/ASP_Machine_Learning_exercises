#Imports:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn.preprocessing
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#1 Feature Engineering ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#(a) Load the Breast Cancer dataset using sklearn.datasets.load_ breast_cancer() as per usual.
breastcancer = load_breast_cancer()
X = breastcancer["data"]
Y = breastcancer["target"]

#(b) Extract polynomial features (without bias!) and interactions up to a degree of 2 using PolynomialFeatures(). How many features do you end up with?
poly = sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)
pf = poly.fit(X, y=Y)
Xt = pf.transform(X)
np.count_nonzero(Xt)
#   Answer: I end up with 279432 features

#(c) Create a pandas.DataFrame() using the polynomials.
#   Use the originally provided feature names to generate names for the polynomials
#   (.get_feature_names() accepts a parameter) and use them as column names.
#   Also add the dependent variable to the object and name the column ”y”.
#   Finally save it as comma-separated textfile named ./output/polynomials.csv
names = pf.get_feature_names(input_features=breastcancer['feature_names'])
df = pd.DataFrame(Xt, columns=names)
df["y"] = breastcancer["target"]
df.to_csv("./output/polynomials.csv", sep=",")
