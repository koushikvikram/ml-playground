"""
Pandas code snippets
"""
from typing import List
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
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
print("DataFrame head:\n{}\n".format(df.head(n=2)))


# 1.2 Series
print("'Make' column values:\n{}\n".format(df.Make))
# print("'Make' column values:\n{}\n".format(df["Make"])) # alternate way

# select multiple columns
print("Multiple columns selected:\n{}\n".format(df[['Make', 'Model', 'MSRP']]))

# adding a column to the DataFrame
df['id']: List = ['nis1', 'hyu1', 'lot2', 'gmc1', 'nis2']
print("DataFrame after adding 'id' column:\n{}\n".format(df))

# changing the contents of a column
df["id"]: List = [1, 2, 3, 4, 5]
print("DataFrame after updating 'id' column:\n{}\n".format(df))

# deleting a column
del df["id"]
print("DataFrame after deleting 'id' column:\n{}\n".format(df))


# 1.3 index - describes how we can access rows from a DataFrame (or a Series)
# two indexes for DataFrame - row, column
# everything that works for Series will also work for Index
print("DataFrame's index:\n{}\n".format(df.index))
print("DataFrame's columns:\n{}\n".format(df.columns))


# 1.4 accessing rows
# accessing rows using their "positional" numbers
print("df.iloc[0]:\n{}\n".format(df.iloc[0]))
print("df.iloc[[2, 3, 0]]:\n{}\n".format(df.iloc[[2, 3, 0]]))

# shuffling the rows in our DataFrame
idx: ndarray = np.arange(5)
np.random.seed(2)
np.random.shuffle(idx)

# assign shuffled df back to df
df: DataFrame = df.iloc[idx]
print("Shuffled DataFrame:\n{}\n".format(df))
print("Indexes of shuffled DataFrame:\n{}\n".format(df.index))

# accessing rows by index rather than position
print("Rows accessed by index of shuffled DataFrame:\n{}\n".format(df.loc[[0, 1]]))

# replace index with default index
print("DataFrame after resetting index to default:\n{}\n".format(df.reset_index(drop=True)))


# 1.5 splitting DataFrame
n_train: int = 3
n_val: int = 1
n_test: int = 1

df_train = df.iloc[:n_train]
df_val = df.iloc[n_train:n_train+n_val]
df_test = df.iloc[n_train+n_val:]

print("Train set:\n{}\n".format(df_train))
print("Validation set:\n{}\n".format(df_val))
print("Test set:\n{}\n".format(df_test))


# ----- 2. operations -----
# 2.1 element-wise operations


# 2.2 filtering


# 2.3 string operations


# 2.4 summarizing operations


# 2.5 missing values


# 2.6 sorting


# 2.7 grouping
