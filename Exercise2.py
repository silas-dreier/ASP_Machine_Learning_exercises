#Imports:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn.preprocessing
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#1 Feature Engineering ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#(a)
breastcancer = load_breast_cancer()
X = breastcancer["data"]
Y = breastcancer["target"]

#(b)
poly = sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)
pf = poly.fit(X, y=Y)
Xt = pf.transform(X)
np.count_nonzero(Xt)
#   Answer: I end up with 279432 features

#(c)
names = pf.get_feature_names(input_features=breastcancer['feature_names'])
df = pd.DataFrame(Xt, columns=names)
df["y"] = breastcancer["target"]
df.to_csv("./output/polynomials.csv", sep=",")

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#clean slate
globals().clear()
#Imports:
import pandas as pd
import seaborn as sns
import sklearn.preprocessing
from sklearn.decomposition import PCA
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#2 Principal Component Analysis::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#(a)
df = pd.read_csv("./data/olympics.csv", index_col=0)
df_desc = df.describe()
#Givent that score is our dependant variable, it would not make much sense to drop it.
# Rather, it is advisable to split the dataset accordingly.
    X = df.iloc[:,0:10]
    Y = df["score"]
colnam=list(X. columns. values)

# (b)
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = pd.DataFrame(scaler.transform(X),columns=colnam)
X_scaled.describe() #assert all varibales have unit variance --> they do

# (c)
pca = PCA(random_state=42)
pca.fit(X_scaled)

sns.heatmap(pca.components_,
            xticklabels=colnam)
#   The variable that loads most prominently on the first component is long.
#   The variable that loads most prominently on the second component is long.
#   The variable that loads most prominently on the third component is 1500.
#   Anwser: This means that the performance in running long holds the most variance and thus serves best as a predictor for
#   our dependent variable score.

#(d)
df = pd.DataFrame(pca.explained_variance_ratio_,
                  columns=["Explained Variance"])
pca.explained_variance_ratio_
df["Cumulative"] = df["Explained Variance"].cumsum()
df.plot(kind="bar")
#   Answer: I need at least 6 components to explain 90% of the variance in the dataset.

