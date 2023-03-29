import pandas as pd
import matplotlib.pyplot as plt

# read the csv file into a pandas DataFrame object
df = pd.read_csv('data.csv', names=['ID', 'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality'])
df = df.apply(pd.to_numeric, errors='coerce')

# drop rows with null values
df.dropna(inplace=True)

# drop duplicates
df.drop_duplicates(inplace=True)

# describe the dataset
print(df.describe())

# compute pairwise correlation of numeric columns only
corr_matrix = df.corr(numeric_only=True)

# print correlation matrix
print(corr_matrix)

# compute mean, median, and mode
print("Mean:\n", df.mean(numeric_only=True))
print("Median:\n", df.median(numeric_only=True))
print("Mode:\n", df.mode(numeric_only=True))

# compute standard deviation and variance of numeric columns only
std_values = df.std(numeric_only=True)
var_values = df.var(numeric_only=True)

# print standard deviation and variance
print("Standard deviation:\n", std_values)
print("Variance:\n", var_values)




# compute summary statistics for individual columns
print("ID column statistics:\n", df['ID'].describe())
print("Fixed acidity column statistics:\n", df['fixed_acidity'].describe())
print("Volatile acidity column statistics:\n", df['volatile_acidity'].describe())
print("Citric acid column statistics:\n", df['citric_acid'].describe())
print("Residual sugar column statistics:\n", df['residual_sugar'].describe())
print("Chlorides column statistics:\n", df['chlorides'].describe())
print("Free sulfur dioxide column statistics:\n", df['free_sulfur_dioxide'].describe())
print("Total sulfur dioxide column statistics:\n", df['total_sulfur_dioxide'].describe())
print("Density column statistics:\n", df['density'].describe())
print("pH column statistics:\n", df['pH'].describe())
print("Sulphates column statistics:\n", df['sulphates'].describe())
print("Alcohol column statistics:\n", df['alcohol'].describe())
print("Quality column statistics:\n", df['quality'].describe())

#Task number 3
#1. Outliers: We can identify potential outliers in the dataset by computing the 
#interquartile range (IQR) and using it to determine the upper and lower limits 
#of the dataset. Any data point that falls outside these limits can be considered
#an outlier. For example, to identify potential outliers in the 'fixed acidity' column,
#we can compute the IQR as follows:

q1 = df['fixed_acidity'].quantile(0.25)
q3 = df['fixed_acidity'].quantile(0.75)
iqr = q3 - q1
lower_limit = q1 - 1.5*iqr
upper_limit = q3 + 1.5*iqr
outliers = df[(df['fixed_acidity'] < lower_limit) | (df['fixed_acidity'] > upper_limit)]
print("Potential outliers in fixed acidity column:\n", outliers)

#2. Distribution: We can visualize the distribution of individual columns using histograms or kernel density plots.
# This can help us identify whether the data is normally distributed, skewed,
#  or has multiple peaks. For example, to visualize the distribution of the 'fixed acidity' column, we can use the following code:
df['fixed_acidity'].plot(kind='hist', bins=20)
plt.xlabel('Fixed Acidity')
plt.show()



#3 Relationships between columns: We can visualize the relationships between pairs of columns using scatterplots or heatmaps. This can help us identify whether there
#  are any linear or nonlinear relationships between the columns. For example, to visualize the relationship between the 'fixed acidity' and 'citric acid' columns,
#  we can use the following code:
plt.scatter(x='fixed_acidity', y='citric_acid', data=df)

# Add labels to the x and y axes
plt.xlabel('Fixed Acidity')
plt.ylabel('Citric Acid')

# Show the plot
plt.show()

#4Data imbalances: We can check whether there are any imbalances in the target variable ('quality') by computing 
# the frequency of each unique value. For example, to check whether the 'quality' variable is balanced or imbalanced, we can use the following code:

quality_counts = df['quality'].value_counts()
print("Frequency of each unique value in quality column:\n", quality_counts)


# Compute pairwise correlation of columns
corr = df.corr()

# Print the correlation matrix
print(corr)