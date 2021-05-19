from glob import glob
import pandas as pd

files = glob("*.csv")
csvs = map(pd.read_csv, files)
dataframe = pd.concat(list(csvs))

dataframe_serie = pd.to_datetime(dataframe["timestamp"])
datetime_index = pd.DatetimeIndex(dataframe_serie.values)

dataframe.set_index(datetime_index, inplace=True)

print(f"Current dataframe shape is {dataframe.shape}")
print(dataframe.describe())
