import numpy as np
import pandas as pd

# Creating numpy arrays
arr = np.array([1, 2, 3, 4, 5])
print("Numpy Array:", arr)  # Output: [1 2 3 4 5]

# Basic numpy array operations
arr_sum = np.sum(arr)
arr_mean = np.mean(arr)
print("Sum of Array:", arr_sum)  # Output: 15
print("Mean of Array:", arr_mean)  # Output: 3.0

# Creating pandas DataFrames
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)
print("DataFrame:\n", df)
# Output:
#    A  B
# 0  1  4
# 1  2  5
# 2  3  6

# DataFrame basic operations
df_sum = df.sum()
df_mean = df.mean()
print("Sum of DataFrame:\n", df_sum)
# Output:
# A     6
# B    15
# dtype: int64
print("Mean of DataFrame:\n", df_mean)
# Output:
# A    2.0
# B    5.0
# dtype: float64

# Selecting columns from a DataFrame
column_a = df['A']
print("Column A:\n", column_a)
# Output:
# 0    1
# 1    2
# 2    3
# Name: A, dtype: int64

# Filtering rows in a DataFrame
filtered_df = df[df['A'] > 1]
print("Filtered DataFrame:\n", filtered_df)
# Output:
#    A  B
# 1  2  5
# 2  3  6

# Handling missing data in pandas
df_with_nan = df.copy()
df_with_nan.loc[1, 'A'] = None
df_filled = df_with_nan.fillna(0)
print("DataFrame with NaN:\n", df_with_nan)
# Output:
#      A  B
# 0  1.0  4
# 1  NaN  5
# 2  3.0  6
print("Filled DataFrame:\n", df_filled)
# Output:
#      A  B
# 0  1.0  4
# 1  0.0  5
# 2  3.0  6

# Grouping data in pandas
grouped_df = df.groupby('A').sum()
print("Grouped DataFrame:\n", grouped_df)
# Output:
#    B
# A   
# 1  4
# 2  5
# 3  6

# Merging and joining DataFrames
df2 = pd.DataFrame({'A': [1, 2, 3], 'C': [7, 8, 9]})
merged_df = pd.merge(df, df2, on='A')
print("Merged DataFrame:\n", merged_df)
# Output:
#    A  B  C
# 0  1  4  7
# 1  2  5  8
# 2  3  6  9

# Pivot tables in pandas
pivot_df = df.pivot_table(values='B', index='A', aggfunc='sum')
print("Pivot Table:\n", pivot_df)
# Output:
#    B
# A   
# 1  4
# 2  5
# 3  6

# DataFrame sorting
sorted_df = df.sort_values(by='A')
print("Sorted DataFrame:\n", sorted_df)
# Output:
#    A  B
# 0  1  4
# 1  2  5
# 2  3  6

# DataFrame aggregation
df_agg = df.agg({'A': 'sum', 'B': 'mean'})
print("Aggregated DataFrame:\n", df_agg)
# Output:
# A    6.0
# B    5.0
# dtype: float64

# Applying functions to DataFrame columns
df['C'] = df['A'].apply(lambda x: x * 2)
print("DataFrame with Applied Function:\n", df)
# Output:
#    A  B  C
# 0  1  4  2
# 1  2  5  4
# 2  3  6  6

# Working with datetime in pandas
df['date'] = pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03'])
df['year'] = df['date'].dt.year
print("DataFrame with Datetime:\n", df)
# Output:
#    A  B  C       date  year
# 0  1  4  2 2021-01-01  2021
# 1  2  5  4 2021-01-02  2021
# 2  3  6  6 2021-01-03  2021

# Plotting data with pandas
# df.plot(x='A', y='B')  # Uncomment to plot

# Saving DataFrames to CSV files
# df.to_csv('output.csv', index=False)  # Uncomment to save

# Creating numpy arrays from lists
arr = np.array([1, 2, 3, 4, 5])
print("Numpy Array from List:", arr)  # Output: [1 2 3 4 5]

# Element-wise operations in numpy
arr_squared = arr ** 2
print("Squared Array:", arr_squared)  # Output: [ 1  4  9 16 25]

# Broadcasting in numpy
arr_broadcast = arr + 10
print("Broadcasted Array:", arr_broadcast)  # Output: [11 12 13 14 15]

# Linear algebra with numpy
matrix = np.array([[1, 2], [3, 4]])
matrix_inv = np.linalg.inv(matrix)
print("Matrix:\n", matrix)
# Output:
# [[1 2]
#  [3 4]]
print("Inverse of Matrix:\n", matrix_inv)
# Output:
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# Random number generation with numpy
random_arr = np.random.rand(5)
print("Random Array:", random_arr)  # Output: [random values]

# Statistical operations with numpy
mean = np.mean(random_arr)
std_dev = np.std(random_arr)
print("Mean of Random Array:", mean)  # Output: [mean value]
print("Standard Deviation of Random Array:", std_dev)  # Output: [std dev value]

# Creating pandas Series
series = pd.Series([1, 2, 3, 4, 5])
print("Pandas Series:\n", series)
# Output:
# 0    1
# 1    2
# 2    3
# 3    4
# 4    5
# dtype: int64

# DataFrame indexing and selection
selected_row = df.loc[0]
selected_cell = df.at[0, 'A']
print("Selected Row:\n", selected_row)
# Output:
# A             1
# B             4
# C             2
# date    2021-01-01 00:00:00
# year        2021
# Name: 0, dtype: object
print("Selected Cell:", selected_cell)  # Output: 1

# DataFrame reshaping (melt, pivot)
melted_df = pd.melt(df, id_vars=['A'], value_vars=['B'])
print("Melted DataFrame:\n", melted_df)
# Output:
#    A variable  value
# 0  1        B      4
# 1  2        B      5
# 2  3        B      6

# Time series analysis with pandas
time_series = pd.date_range(start='1/1/2021', periods=5, freq='D')
ts_df = pd.DataFrame({'date': time_series, 'value': np.random.rand(5)})
ts_df.set_index('date', inplace=True)
print("Time Series DataFrame:\n", ts_df)
# Output:
#                 value
# date                 
# 2021-01-01  0.123456
# 2021-01-02  0.234567
# 2021-01-03  0.345678
# 2021-01-04  0.456789
# 2021-01-05  0.567890