#Imports:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


#1.Tips
#(a)
df = pd.DataFrame(sns.load_dataset("tips"))

#(b)
df["day"] = df["day"].replace(to_replace=["Thur", "Fri", "Sat", "Sun"], value=["Thursday", "Friday", "Saturday", "Sunday"])

#c)
g = sns.FacetGrid(df, col="sex", palette="colorblind", hue="day")
g.map(sns.scatterplot, "total_bill", "tip")
g.add_legend()
g.set_axis_labels("Total Bill in $", "Tips in $")
g.savefig("./output/tips.pdf")

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
#"There are 21 occupations present."
print(OccuCount.head(1)) #"Given that .value_counts() sorts in an descending order per default, the most common occupation must be student

#(f)
OccuCount=OccuCount.reset_index()
OccuCount.sort_index(inplace=True)

fig, ax = plt.subplots(figsize=(20, 5))
OccuCount.plot.bar(x="index", y="occupation",ax=ax)
ax.set(ylabel="Count", xlabel="Occupations")
fig.subplots_adjust(bottom=0.3)
fig.savefig("./output/occupations.pdf")

#In this case, the figure would look exactly the same if it would not have been ordered by index,
#because I reset the index based on the order that was created by .value_counts()

#3
#a)
FNAME = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df_3 = pd.read_csv(FNAME, names=["sepal length (in cm)" , "sepal width (in cm)" , "petal length (in cm)" , "petal width (in cm)" , "class"])

#b)
df_3.loc[10:29, 'petal length (in cm)'] = np.nan

#c)
df_3['petal length (in cm)']= df_3['petal length (in cm)'].fillna(1.0)

#d)
df_3.to_csv("./output/iris.csv", sep=",", index=False)

#e)
g = sns.FacetGrid(df_3, row="class", palette="colorblind", hue="class", aspect=3, height=4)
g.map_dataframe(sns.violinplot, args=["sepal length (in cm)" , "sepal width (in cm)" , "petal length (in cm)" , "petal width (in cm)"])
g.set_xticklabels(rotation = 45)
g.figure.subplots_adjust(bottom=0.15)
g.savefig("./output/iris.pdf")

#4
#a)
FNAME = "https://query.data.world/s/wsjbxdqhw6z6izgdxijv5p2lfqh7gx
df_4=pd.read_csv(FNAME, sep=",", low_memory=False)

#b)
df_4.info() #estimates memory usage
df_4.info(memory_usage="deep") #shows actual memory usage
#The Dataframe requires 859.5MB in memory usage

#c)
df_4_copy=df_4.select_dtypes(include=['object'])

#d) Taking "very few" to mean "below 50%" due to the subsequent taks
d4cd = df_4_copy.describe(include=['object'])
d4cd.loc['ratio'] = d4cd.loc['unique'] / d4cd.loc['count']
d4cdn=d4cd.drop(['count', 'unique', 'freq', 'top'])
d4cdm=d4cdn.melt()
very_few = d4cdm.loc[d4cdm['value'] <= 0.5]
#Provded that "very few" means below 50%, the 76 variables found in very_few have very few unique values
#compared to the number of observations


#e) No,as that would not result in something useful. Of course, conversion reduces size, thus saving memory and creating speed.
#It can also be useful when plotting categorical graphs.
#However, if there are more than 50% non-unique values, costs may outweigh the benefits, as it would not make much sense.
#The more categories there are, the less useful is categorisation (ordering). This may result in problems, especially
#when trying to plot appropriate diagrams.

#f)
CatVar = very_few['variable']
df_4[CatVar] = df_4[CatVar].astype("category")

#g)
df_4.info(memory_usage="deep")
#The Dataframe requires 168.9MB in memory usage

#h)
#We could run the following code:
#df_4=pd.read_csv("./game_logs.csv", sep=",", low_memory=True, dtype={ }
#with the dtype for all columns specified as "category", provided that we would have a list of all of them.
#Specifiying them by hand would be quite cumbersome

#i)
df_4_subset = df_4._get_numeric_data()
df_4_subset.to_csv("./output/WS_Subset.csv")
#The File has almost 52MB
df_4_subset.to_feather("./output/WS_Subset.feather")
#The File has about 31MB

