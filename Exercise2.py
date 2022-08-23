#Imports:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn.datasets import load_breast_cancer
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

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#clean slate
globals().clear()
#Imports:
import pandas as pd
import seaborn as sns
import sklearn.preprocessing
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#3 Principal Component Analysis::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#(a)
iris = sklearn.datasets.load_iris()
    X = iris["data"]
    Y = iris["target"]

#(b)
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = pd.DataFrame(scaler.transform(X))
X_scaled.describe() #--> test: standard errors euqal? Yes, thus unit variance achieved

#kmeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
kmeans.labels_

#Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3)
agg.fit(X)

#dbscan
dbscan = DBSCAN(eps=1, min_samples=2)
dbscan.fit(X)

#Combine
df = pd.DataFrame()
df["kmeans"] = kmeans.labels_
df["agg"] = agg.labels_
df["dbscan"] = dbscan.labels_
df["dbscan"].value_counts()

#(d)
print(silhouette_score(X, kmeans.labels_))
print(silhouette_score(X, agg.labels_))
print(silhouette_score(X, dbscan.labels_)) #highest score
#   Noise assignments from DBSCAN have to be treated differently
#   because, in contrast to the other two models, DBSCAN declares
#   data as noise if it is far away from the clusters, rather than
#   trying to group everything into the clusters

#(e)
temp = pd.DataFrame(X, columns=iris["feature_names"])
df["sepal width (cm)"] = temp["sepal width (cm)"]
df["petal length (cm)"] = temp["petal length (cm)"]

#(f)
df.loc[df["dbscan"] == 0, "dbscan"] = "Noise"

#(g)
#preapre data for plotting
df1 = pd.melt(df, id_vars=["sepal width (cm)", "petal length (cm)"])
df1.rename(columns={'variable': 'cluster algorithm', 'value': 'cluster assignments'}, inplace=True)

#plot
fig = sns.relplot(
    data=df1, x="sepal width (cm)", y="petal length (cm)",
    col="cluster algorithm", hue="cluster assignments", palette="colorblind",
    kind="scatter"
)
fig.savefig("./output/cluster_petal.pdf")
#The noise assignment does make sense. Since the other methods tried to find clusters, they also tried grouping that part of
# the data into one cluster, that dbscan in a certain way "clustered" as noise. Naturally, they then came up with three clusters
# where one describes the Noise, while dbscan grouped the two clusters from the other two methods into one and label the the "third cluster"
# as noise, so to speak.