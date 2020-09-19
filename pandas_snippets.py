"""
Pandas code snippets
"""
from typing import List
import pandas as pd
import numpy as np
from pandas import DataFrame, Series, Index
from numpy import ndarray


# ----- 1. absolute basics -----
data: List[List] = [
    ['Nissan', 'Stanza', 1991, 138, 4, 'MANUAL', 'sedan', 2000],
    ['Hyundai', 'Sonata', 2017, None, 4, 'AUTOMATIC', 'Sedan', 27150],
    ['Lotus', 'Elise', 2010, 218, 4, 'MANUAL', 'convertible', 54990],
    ['GMC', 'Acadia',  2017, 194, 4, 'AUTOMATIC', '4dr SUV', 34450],
    ['Nissan', 'Frontier', 2017, 261, 6, 'MANUAL', 'Pickup', 32340],
]

columns: List = [
    'Make', 'Model', 'Year', 'Engine HP', 'Engine Cylinders',
    'Transmission Type', 'Vehicle_Style', 'MSRP'
]

# 1.1 DataFrame
# creating a DataFrame from a list of lists
df: DataFrame = pd.DataFrame(data, columns=columns)
print("DataFrame head:\n{}\n"
      .format(df.head(n=2)))


# 1.2 Series
print("'Make' column values:\n{}\n"
      .format(df.Make))
# print("'Make' column values:\n{}\n".format(df["Make"])) # alternate way

# select multiple columns
print("Multiple columns selected:\n{}\n"
      .format(df[['Make', 'Model', 'MSRP']]))

# adding a column to the DataFrame
df['id']: List = ['nis1', 'hyu1', 'lot2', 'gmc1', 'nis2']
print("DataFrame after adding 'id' column:\n{}\n"
      .format(df))

# changing the contents of a column
df["id"]: List = [1, 2, 3, 4, 5]
print("DataFrame after updating 'id' column:\n{}\n"
      .format(df))

# deleting a column
del df["id"]
print("DataFrame after deleting 'id' column:\n{}\n"
      .format(df))


# 1.3 index - describes how we can access rows from a DataFrame (or a Series)
# two indexes for DataFrame - row, column
# everything that works for Series will also work for Index
print("DataFrame's index:\n{}\n"
      .format(df.index))
print("DataFrame's columns:\n{}\n"
      .format(df.columns))


# 1.4 accessing rows
# accessing rows using their "positional" numbers
print("df.iloc[0]:\n{}\n"
      .format(df.iloc[0]))
print("df.iloc[[2, 3, 0]]:\n{}\n"
      .format(df.iloc[[2, 3, 0]]))

# shuffling the rows in our DataFrame
idx: ndarray = np.arange(5)
np.random.seed(2)
np.random.shuffle(idx)

# assign shuffled df back to df
df: DataFrame = df.iloc[idx]
print("Shuffled DataFrame:\n{}\n"
      .format(df))
print("Indexes of shuffled DataFrame:\n{}\n"
      .format(df.index))

# accessing rows by index rather than position
print("Rows accessed by index of shuffled DataFrame:\n{}\n"
      .format(df.loc[[0, 1]]))

# replace index with default index
print("DataFrame after resetting index to default:\n{}\n"
      .format(df.reset_index(drop=True)))


# 1.5 splitting DataFrame
n_train: int = 3
n_val: int = 1
n_test: int = 1

df_train: DataFrame = df.iloc[:n_train]
df_val: DataFrame = df.iloc[n_train:n_train+n_val]
df_test: DataFrame = df.iloc[n_train+n_val:]

print("Train set:\n{}\n"
      .format(df_train))
print("Validation set:\n{}\n"
      .format(df_val))
print("Test set:\n{}\n"
      .format(df_test))

print("DataFrame indexed with a list of booleans:\n{}\n"
      .format(df[[True, True, False, False, True]]))


# ----- 2. operations -----
# 2.1 element-wise operations
# Series supports element-wise operations
# operation applied to each element in the series -> returns Series
print("Engine HP doubled:\n{}\n"
      .format(df['Engine HP']*2))
print("After 2000:\n{}\n"
      .format(df['Year'] > 2000))
print("Nissan after 2000:\n{}\n"
      .format((df['Make'] == "Nissan") & (df['Year'] > 2000)))


# 2.2 filtering
print("Details of Nissan cars:\n{}\n"
      .format(df[df['Make'] == 'Nissan']))
print("Automatic transmission cars made after 2010:\n{}\n"
      .format(df[(df['Transmission Type'] == 'AUTOMATIC') & (df['Year'] > 2010)]))


# 2.3 string operations
print("Vehicle Style Lower case:\n{}\n"
      .format(df['Vehicle_Style'].str.lower()))
print("Vehicle Style Normalized:\n{}\n"
      .format(df['Vehicle_Style'].str.lower().str.replace(' ', '_')))

df.columns: Series = df.columns.str.lower().str.replace(' ', '_')
print("Column Names Normalized:\n{}\n"
      .format(df.columns))

print("Only string columns:\n{}\n"
      .format(df.dtypes[df.dtypes == 'object'].index))

# normalize each string column
string_columns: Index = df.dtypes[df.dtypes == 'object'].index
print(type(string_columns))
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')
print("String-Normalized DataFrame:\n{}\n"
      .format(df))


# 2.4 summarizing operations
# useful for Exploratory Data Analysis
print("Mean price:\n{}\n"
      .format(df.msrp.mean()))
print("MSRP description:\n{}\n"
      .format(df.msrp.describe()))
print("Mean of all columns:\n{}\n"
      .format(df.mean()))
print("Description of all columns, rounded to 2 decimals:\n{}\n"
      .format(df.describe().round(2)))


# 2.5 missing values
print("Missing values:\n{}\n"
      .format(df.isnull()))
print("Missing value count by column:\n{}\n"
      .format(df.isnull().sum()))
# replace missing values in engine_hp with null
print("Replacing missing value of 'engine_hp' with 0:\n{}\n"
      .format(df.engine_hp.fillna(0)))

df.engine_hp: Series = df.engine_hp.fillna(df.engine_hp.mean())
print("Replacing missing value of 'engine_hp' with mean:\n{}\n"
      .format(df.engine_hp))


# 2.6 sorting
print("Sort DataFrame by msrp:\n{}\n"
      .format(df.sort_values(by="msrp")))
print("Sort DataFrame by msrp in descending order:\n{}\n"
      .format(df.sort_values(by="msrp", ascending=False)))


# 2.7 grouping
print("Mean msrp for each 'transmission_type':\n{}\n"
      .format(df.groupby('transmission_type').msrp.mean()))
print("Mean of msrp and count for each 'transmission_type':\n{}\n"
      .format(df.groupby('transmission_type').msrp.agg(['mean', 'count'])))
