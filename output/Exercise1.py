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
g.savefig("tips.pdf")

#2. Occupations
#(a)
FNAME = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user"
df_2 = pd.read_csv(FNAME, sep="|")

#(b)
print(df_2.tail(10))
print(df_2.head(25))



#(d)
OccuCount = df_2["occupation"].value_counts(dropna=False)

#(e)
OCC=OccuCount.count()
print(f"There are {OCC} occupations present.")
print("Given that .value_counts() sorts in an descending order per default, the most common occupation must be")
print(OccuCount.head(1))
