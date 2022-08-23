#Imports::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.linear_model import LogisticRegression as Logit
from sklearn.linear_model import LinearRegression as OLS
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#1  Exercises for Supervised Machine Learning
#(a)
df = pd.read_csv("./output/polynomials.csv", index_col=0)

#(b)
y = df["y"]
X = df. loc[:, df. columns != "y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#(c)
lm = OLS().fit(X_train, y_train)
ridge = Ridge(alpha=0.3).fit(X_train, y_train)
lasso = Lasso(alpha=0.3).fit(X_train, y_train)
lm.score(X_test, y_test)
ridge.score(X_test, y_test)
lasso.score(X_test, y_test)
#Answer: Lasso yields the best predicition (R-Squared of 0.75)

#(d)
colnam=list(X.columns.values)
df = pd.DataFrame(lm.coef_, index=colnam, columns=["OLS"])
df["Lasso"] = lasso.coef_
df["Ridge"] = ridge.coef_
df.query("Lasso == 0 & Ridge != 0").shape[0]
#The Lasso coefficients are equal to zero in 449 cases where the ridge coefficients are not equal to zero

#(e)
#Plot Data
df1 = df
df1["OLS"]   = lm.coef_.cumsum()
df1["Lasso"] = lasso.coef_.cumsum()
df1["Ridge"] = ridge.coef_.cumsum()
df1.plot.bar()
plt.barh(height=10, width=30, data=df1)

plt.barh(Product,Quantity)


#(e) Using matplotlib.pyplot, create a horizontal bar plot of dimension 10x30 showing the coefficient sizes.
#Save the figure as ./output/polynomials.pdf.
