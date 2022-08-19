import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt


#1.Tips
#(a)
df = pd.DataFrame(sns.load_dataset("tips"))

#(b)
df["day"] = df["day"].replace(to_replace=["Thur", "Fri", "Sat", "Sun"], value=["Thursday", "Friday", "Saturday", "Sunday"])

#c)
g = sns.FacetGrid(df, col="sex", hue="day")
g.map(sns.scatterplot, "total_bill", "tip")
g.add_legend()
g.set_axis_labels("Total Bill in $", "Tips in $")
g.savefig("C:/Users/silas/PyCharmProjects/ASP_Machine_Learning_exercises/output/tips.pdf")

#2. Occupations
#(a)
FNAME = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user"
df_2 = pd.read_csv(FNAME, sep="|")

#(b)
print(df_2.tail(10))
print(df_2.head(25))

#(c)
df_2.info(memory_usage=True)
print(' "user_id" and "age" are Integers, "gender", "occupation" and "zip_code" are objects.')

#(d)
OccuCount = pd.DataFrame(df_2["occupation"].value_counts(dropna=False))

#(e)
OCC=OccuCount.count()
print("There are 21 occupations present.")
print("Given that .value_counts() sorts in an descending order per default, the most common occupation must be student, as")
print(OccuCount.head(1))

#(f)
OccuCount=OccuCount.reset_index()
OccuCount.sort_index(inplace=True)

fig, ax = plt.subplots(figsize=(20, 5))
OccuCount.plot.bar(x="index", y="occupation",ax=ax)
ax.set(ylabel="Count", xlabel="Occupations")
fig.savefig("C:/Users/silas/PyCharmProjects/ASP_Machine_Learning_exercises/output/occupations.pdf")

#In this case, the figure would look exactly the same if it would not have been ordered by index,
#because I reset the index based on the order that was created by .value_counts()

#3
#a)
FNAME = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df_3 = pd.read_csv(FNAME, names=["sepal length (in cm)" , "sepal width (in cm)" , "petal length (in cm)" , "petal width (in cm)" , "class"])
df_3.info()

#b)
df_3.loc[10:29, 'petal length (in cm)'] = "NA"

#c)
df_3= df_3.fillna(1.0)

#d)
df_3.to_csv("C:/Users/silas/PyCharmProjects/ASP_Machine_Learning_exercises/output/iris.csv", sep=",", index=False)
