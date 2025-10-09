import pandas as pd
df=pd.read_csv("iris.csv")
print(df.head())#print first 5 rows
print(df.tail())#print last 5 rows
print(df.dtypes)
print(df.index)
print(df.columns)
print(df.values)