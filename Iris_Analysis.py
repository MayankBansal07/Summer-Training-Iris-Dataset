import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("Iris.csv")
print("Data loaded successfully.")
print(data)
print("-------------------------------")

print("Data Information:")
print(data.info())
print("-------------------------------")

print("Statistical Summary:")
print(data.describe())
print("-------------------------------")

print("First 5 Rows:")
print(data.head())
print("-------------------------------")

print("Last 5 Rows:")
print(data.tail())
print("-------------------------------")

print("Column Names:")
print(data.columns)
print("-------------------------------")

print("Column in Vertical Format:")
for column in data.columns:
    print(column)
print("-------------------------------")

print("Shape of the DataFrame:")
print(data.shape)
print("-------------------------------")

print("No. of Rows")
print(data.shape[0])
print("-------------------------------")

print("No. of Columns")
print(data.shape[1])
print("-------------------------------")

print("Count of each column:")
print(data.count())
print("-------------------------------")

print("Statical Functions:")
print("Mean: ",data.mean(numeric_only=True))
print("Median: ",data.median(numeric_only=True))
print("Mode: ",data.mode(numeric_only=True))
print("Standard Deviation: ",data.std(numeric_only=True))
print("Variance: ",data.var(numeric_only=True))
print("Minimum: ",data.min(numeric_only=True))
print("Maximum: ",data.max(numeric_only=True))
print("-------------------------------")

print("No. of unique values")
print(data.nunique())
print("-------------------------------")

print("Unique value in a column:")
print(data['Species'].unique())
print("--------------------------------")

print("Rename a column:")
data.rename(columns={'SepalLengthCm': 'SepalLengthCm'})
print("-------------------------------")

print("Check for missing values:")
print(data.isnull().sum())
print("--------------------------------")

print("Total of duplicated values:")
print(data.duplicated().sum())
print("--------------------------------")

print("Count of each species:")
sns.countplot(x='Species', data=data, palette="viridis", hue='Species')
plt.title("Species Distribution")
plt.show()
print("----------------------------------------------")

print("Boxplot showing outliers in all features using boxplot:")
sns.boxplot(data=data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], palette="viridis")
plt.title("Boxplot of Sepal and Petal Features")
plt.ylabel("Measurement (cm)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Outliers found in this boxplot are:
# sepal width - 4
print("----------------------------------------------")

print("Visualizing how species are distributed across the Sepal Length:")
sns.boxplot(x='Species', y='SepalLengthCm',data=data)
plt.title("Boxplot of Sepal length by Species")
plt.show()
print("----------------------------------------------")

print("Voilin Plot showing distribution of Petal Length by Species:")
sns.violinplot(x='Species', y='PetalLengthCm', data=data)
plt.title("Violin Plot of Petal Length by Species")
plt.show()
print("----------------------------------------------")

print("Scatter Plot of Sepal Length vs Petal Length:")
sns.scatterplot(x='SepalLengthCm', y='PetalLengthCm',hue='Species', data=data)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.title("Scatter Plot of Sepal Length vs Petal Length")
plt.show()
print("----------------------------------------------")

print("Scatter Plot of Sepal Width vs Petal Width:")
sns.scatterplot(x='SepalWidthCm', y='PetalWidthCm', hue='Species', data=data)
plt.title('Petal Length vs Petal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.show()

print("Pairplot of all features colored by Species:")
sns.pairplot(data, hue='Species', height=2)
plt.suptitle("Pairplot of Iris Dataset Features")
plt.show()
print("----------------------------------------------")

print("Correlation Matrix:")
numerical_data = data.select_dtypes(include=['number'])
correlation_matrix = numerical_data.corr()
print(correlation_matrix)
print("----------------------------------------------")

print("Visualizing Correlation Matrix using Heatmap:")
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()
print("----------------------------------------------")

print("Finding Outliers in Sepal Width using IQR method:")
Q1 = data['SepalWidthCm'].quantile(0.25)
Q3 = data['SepalWidthCm'].quantile(0.75)
IQR = Q3 - Q1
Lower_limit = Q1 - 1.5 * IQR
Upper_limit = Q3 + 1.5 * IQR
outliers = data[(data['SepalWidthCm'] < Lower_limit) | (data['SepalWidthCm'] > Upper_limit)]
print("Outliers in Sepal Width:")
print(outliers)
print("----------------------------------------------")

print("Z-Score Method for Outlier Detection:")
from scipy.stats import zscore
z_scores = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].apply(zscore)
print("Z-Scores:\n", z_scores)
outliers_z = data[(z_scores.abs() > 3).any(axis=1)]
print("Outliers using Z-Score method:")
print(outliers_z)
print("----------------------------------------------")

print("Removig Outliers from Data:")
upper = np.where(data['SepalWidthCm'] >= (Q3+1.5*IQR))
lower = np.where(data['SepalWidthCm'] <= (Q1-1.5*IQR))
print("Shape before removing outliers from data: ",data.shape)
data.drop(upper[0], inplace=True)
data.drop(lower[0], inplace=True)
print("Shape after removing outliers from data: ",data.shape)
print("----------------------------------------------")