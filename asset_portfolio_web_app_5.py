# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime,timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
import statsmodels.api as sm
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from statsmodels.tools.sm_exceptions import ValueWarning, HessianInversionWarning, ConvergenceWarning
import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline


#Read the data

df = pd.read_csv("https://raw.githubusercontent.com/Winnie-Mukunzi/Dissertation_updated/main/PensionAssets_30112023.csv")
#df.head()

#Data attributes and records

#df.shape

#Compute Return on Investment (ROI)

# Assuming df is your DataFrame
df['ROI-Demand DEPs'] = df['Cash, Demand & Fixed Deposits - Income'] / df['Cash, Demand & Fixed Deposits']
df['ROI-C & C Bonds'] = df['Commercial Paper and Corporate Bonds - Income'] / df['Commercial Paper and Corporate Bonds']
df['ROI-Fixed DEPs'] = df['Fixed and Time Deposits - Income'] / df['Fixed and Time Deposits']
df['ROI-GFs'] = df['Guaranteed Funds - Income'] / df['Guaranteed Funds']
df['ROI-IProperty'] = df['Immovable Property - Income'] / df['Immovable Property']
df['ROI-Government SECs'] = df['Kenya Government Securities - Income'] / df['Kenya Government Securities']
df['ROI-Offshore INVE'] = df['Offshore Investments - Income'] / df['Offshore Investments']
df['ROI-P Unit Trusts'] = df['Property Unit Trusts - Income'] / df['Property Unit Trusts']
df['ROI-Qouted EQ'] = df['Quoted Equity - Income'] / df['Quoted Equity']
df['ROI-UNQouted EQ'] = df['Unquoted Equity - Income'] / df['Unquoted Equity']

#df.tail()

#df.columns

cdf = df[['Period', 'ROI-Demand DEPs',
       'ROI-C & C Bonds', 'ROI-Fixed DEPs', 'ROI-GFs', 'ROI-IProperty',
       'ROI-Government SECs', 'ROI-Offshore INVE', 'ROI-P Unit Trusts',
       'ROI-Qouted EQ', 'ROI-UNQouted EQ' ]]
#cdf.head()

# Convert 'Period' column to datetime format
cdf['Period'] = pd.to_datetime(cdf['Period'], format='%b-%y')

# Set 'Period' as the index
cdf.set_index('Period', inplace=True)

#cdf.columns

cdf = cdf.sort_index(ascending=True)
#cdf.head()

#Compute mean for each period and exclude Missing values from the computations for ach of the asset class.

# Columns to compute mean for
columns_to_mean = ['ROI-Demand DEPs', 'ROI-C & C Bonds', 'ROI-Fixed DEPs',
                   'ROI-GFs', 'ROI-IProperty', 'ROI-Government SECs',
                   'ROI-Offshore INVE', 'ROI-P Unit Trusts', 'ROI-Qouted EQ',
                   'ROI-UNQouted EQ']

# Compute mean for each period without counting missing values
cdf_2 = cdf[columns_to_mean].resample('m').mean()

pd.set_option('display.float_format', '{:.2f}'.format)



#Missing values

cdf_2 = cdf_2.fillna(method='ffill')
#cdf_2.head()

# Replace NaN values with the mean of each column
cdf_2 = cdf_2.fillna(cdf_2.mean())

# Replace negative values with the mean of each column
for column in cdf_2.columns:
    feature_mean = cdf_2[column].mean()
    cdf_2[column] = np.where((cdf_2[column].isnull()) | (cdf_2[column] < 0), feature_mean, cdf_2[column])

#cdf_2.head()

#Cleaning inf values

# 'Period' being index column
columns_to_process = ['ROI-Demand DEPs', 'ROI-C & C Bonds', 'ROI-Fixed DEPs',
                       'ROI-GFs', 'ROI-IProperty', 'ROI-Government SECs',
                       'ROI-Offshore INVE', 'ROI-P Unit Trusts', 'ROI-Qouted EQ',
                       'ROI-UNQouted EQ']

# Replace 'inf' with NaN
cdf_2.replace([np.inf, -np.inf], np.nan, inplace=True)

# Replace NaN values with the mean of each column
mean_values = cdf_2[columns_to_process].mean()
cdf_2[columns_to_process] = cdf_2[columns_to_process].fillna(mean_values)

pd.set_option('display.float_format', '{:.2f}'.format)

#cdf_2.head()

#Clean outliers

# List of columns to process
columns_to_process = ['ROI-Demand DEPs', 'ROI-C & C Bonds',
                       'ROI-GFs', 'ROI-IProperty', 'ROI-Government SECs',
                       'ROI-Offshore INVE', 'ROI-P Unit Trusts', 'ROI-Qouted EQ',
                       'ROI-UNQouted EQ']

# Replace values greater than 1 with NAN
cdf_2[columns_to_process] = np.where(cdf_2[columns_to_process] > 0.3, np.nan, cdf_2[columns_to_process])

# Replace NaN values with the mean of each column
mean_values = cdf_2[columns_to_process].mean()
cdf_2[columns_to_process] = cdf_2[columns_to_process].fillna(mean_values)

cdf_3 = cdf_2.drop('ROI-Fixed DEPs', axis=1)

# CBR_Data

# Monthly Central Bank Rate Data


# Load the data
cbk_rate = pd.read_csv("https://raw.githubusercontent.com/Winnie-Mukunzi/Module-IV/main/Central_Bank_Rate_data.csv")

# Convert 'Date' column to datetime format
cbk_rate['Date'] = pd.to_datetime(cbk_rate['Date'], format='%y-%m-%d')

# Set the 'Date' column as the index
cbk_rate.set_index('Date', inplace=True)

# Group by month and take the mean rate for each month
monthly_cbk_rate = cbk_rate.groupby(pd.Grouper(freq='M')).mean()

# Create a complete date range with all months included
complete_date_range = pd.date_range(start=monthly_cbk_rate.index.min(), end=monthly_cbk_rate.index.max(), freq='M')

# Reindex the DataFrame to include the complete date range
monthly_cbk_rate = monthly_cbk_rate.reindex(complete_date_range)

# Forward fill missing values within each group
monthly_cbk_rate = monthly_cbk_rate.ffill()

# Adjust the date to represent the end of each month
monthly_cbk_rate.index = monthly_cbk_rate.index + pd.offsets.MonthEnd(0)

# Filter data to include only dates between January 2013 and December 2023
cbk_rate_final = monthly_cbk_rate['2013-01':'2023-12']

# Display the updated DataFrame
#print(cbk_rate_final)

#Visualizations

#Visualize all the asset classes in a time series plot.

# List of columns to plot
columns_to_plot = ['ROI-C & C Bonds','ROI-GFs','ROI-IProperty', 'ROI-Government SECs', 'ROI-Offshore INVE',
                   'ROI-P Unit Trusts', 'ROI-Qouted EQ']

# Set a larger figure size
plt.figure(figsize=(14, 6))

# Use the 'tab10' colormap for distinct colors
colors = plt.cm.tab10(np.linspace(0, 1, len(columns_to_plot)))

# Plot each column with a different color
for column, color in zip(columns_to_plot, colors):
    plt.plot(cdf_3.index, cdf_3[column], label=column, color=color)

    # Annotate the last value of each line inside the graph
    plt.annotate(f'{column}', xy=(cdf_2.index[-1], cdf_3[column].iloc[-1]),
                 xytext=(5, 5), textcoords='offset points', color=color)

# Set labels and title
plt.xlabel('Date')
plt.ylabel('ROI')
plt.title('ROI Time Series for Each Asset')
plt.grid()
plt.show()

# Describe the cleaned data set

cdf_3.describe()

# Bar gragh of mean vs standard deviation of the assets

# Select only mean and std rows
mean_variance_summary = cdf_3.describe().loc[['mean', 'std']]

# Transpose for easier plotting
mean_variance_summary = mean_variance_summary.transpose()

# Sort by mean in descending order
mean_variance_summary_sorted = mean_variance_summary.sort_values(by='mean', ascending=False)

# Plot as a bar graph
mean_variance_summary_sorted.plot(kind='bar', figsize=(16, 4))
plt.xlabel('Asset Classes')
plt.ylabel('Values')
plt.title('Mean - Variance comparison of the Asset Classes')
plt.legend(['Mean', 'Standard Deviation'])
plt.show()

#Plot a indifference curve

# Select only mean and std rows
mean_variance_summary = cdf_3.describe().loc[['mean', 'std']]

# Transpose for easier plotting
mean_variance_summary = mean_variance_summary.transpose()

# Plot mean against standard deviation
plt.figure(figsize=(16, 4))
plt.scatter(mean_variance_summary['std'], mean_variance_summary['mean'], color='blue')
plt.xlabel('Standard Deviation')
plt.ylabel('Mean')
plt.title('Indifference Curve: Mean vs Standard Deviation of Asset Classes')

# Annotate each point with the asset class
for i, txt in enumerate(mean_variance_summary.index):
    plt.annotate(txt, (mean_variance_summary['std'].iloc[i], mean_variance_summary['mean'].iloc[i]), textcoords="offset points", xytext=(0,5), ha='center')

plt.grid(True)
plt.show()

#Relationship among asset classes

#Covariance

correlation_matrix = cdf_3.corr()

# Plot a heatmap of the correlation matrix

plt.figure(figsize=(12, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix among the asset classes')
plt.show()

# There seems to be postive relationship between Commercial paper/ corporate bond with Unquoted equity and Offshore investments.

# There is also negative relationship between guarenteed funds and government securities.

#Model development

##Markowitz Mean Variance (MV) Portfolio Optimization

#Markowitz Portfolio Optimization relies on the principle of balancing risk and return by strategically allocating assets. By considering the correlation and variance among different investments, the optimization process aims to construct a portfolio that achieves the highest possible return for a given level of risk or, conversely, the lowest possible risk for a targeted level of return. This methodology emphasizes the importance of not putting all investment eggs in one basket, promoting diversification as a means to spread risk across different asset classes.


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

# Functions to select assets based on correlation and volatility
def absHighPass(cdf_3, absThresh):
    c = cdf_3.columns.values
    a = np.abs(cdf_3.values)
    np.fill_diagonal(a, 0)
    i = np.where(a >= absThresh)[0]
    i = sorted(i)
    return cdf_3.loc[c[i],c[i]]

def absHigh(cdf_3, num):
    c = cdf_3.columns.values
    a = np.abs(cdf_3.values)
    np.fill_diagonal(a, 0)
    i = (-a).argpartition(num, axis=None)[:num]
    i, _ = np.unravel_index(i, a.shape)
    i = sorted(i)
    return cdf_3.loc[c[i],c[i]]

def selLow(cdf_3, num):
    c = cdf_3.columns.values
    a = cdf_3.values
    np.fill_diagonal(a, 0)
    i = (a).argpartition(num, axis=None)[:num]
    i, _ = np.unravel_index(i, a.shape)
    i = sorted(i)
    return cdf_3.loc[c[i],c[i]]

# Calculate average returns for each asset
ra = np.mean(cdf_3, axis=0)

# Create a covariance matrix
covar = cdf_3.cov()

# Calculate annualized volatility for each asset
vols = np.sqrt(np.diagonal(covar))

# Create weights array
num_assets = len(ra)
weights = np.concatenate([np.linspace(start=2, stop=1, num=num_assets),
                          np.zeros(1600),
                          np.linspace(start=-1, stop=-2, num=num_assets)])

# Ensure weights array has the same length as the number of assets
weights = weights[:num_assets]

# Normalize weights to ensure they sum up to 1
weights /= weights.sum()

# Calculate Sharpe Ratio for each asset
sharpe_ratios = ra / vols

# Calculate portfolio mean return
mean_return_mv = np.dot(weights, ra)

# Calculate Sharpe Ratio of the portfolio
sharpe_ratio = np.dot(weights, sharpe_ratios)

# Create a DataFrame to store results
result_df_MV = pd.DataFrame({
    'Asset': ra.index,
    'MV_Weight': weights
})

# Display the result DataFrame
#print(result_df_MV)

#print("Portfolio Mean Return:", mean_return_mv)

# Display the Sharpe Ratio of the portfolio
#print("Portfolio Sharpe Ratio:", sharpe_ratio)

## K-Means Clustering

from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

# Set the originalRows, originalColumns
originalRows = 120   # excluding header
originalColumns = 9  # excluding date

# Extract asset labels
assetLabels = cdf_3.columns[1:originalColumns+1].tolist()

# Extract stock prices excluding header and trading dates
dfAssets_returns = cdf_3.iloc[:, 1:]  # Adjusted to include all rows

# Compute mean returns and variance-covariance matrix of returns
meanReturns = dfAssets_returns.mean()
covReturns = dfAssets_returns.cov()

# Prepare asset parameters for k-means clustering
assetParameters = np.column_stack((meanReturns, covReturns))

# Set the range of clusters to test for hyperparameter tuning
param_grid = {'n_clusters': range(1, 10)}

# Perform hyperparameter tuning using GridSearchCV
kmeans_grid = GridSearchCV(KMeans(), param_grid, cv=5)
kmeans_grid.fit(assetParameters)

# Get the best number of clusters from hyperparameter tuning
best_clusters = kmeans_grid.best_params_['n_clusters']

# Perform K-means clustering with the best number of clusters
assetsCluster = KMeans(n_clusters=best_clusters, algorithm='elkan', max_iter=600)
labels = assetsCluster.fit_predict(assetParameters)

# Initialize a dictionary to store assets in each cluster
cluster_assets = {i: [] for i in range(best_clusters)}

# Assign assets to clusters
for i, label in enumerate(labels):
    cluster_assets[label].append(assetLabels[i])

# Initialize a list to store DataFrame for each cluster
cluster_dfs = []

# Iterate over clusters and allocate weights for each asset
for cluster_num, assets_in_cluster in cluster_assets.items():
    # Select data for the current cluster
    cluster_data = dfAssets_returns[assets_in_cluster]

    # Calculate mean returns, covariance matrix, and volatility for the current cluster
    ra_cluster = cluster_data.mean(axis=0)
    covar_cluster = cluster_data.cov()
    vols_cluster = np.sqrt(np.diagonal(covar_cluster))

    # Calculate weights using mean-variance optimization
    inv_covariance = np.linalg.inv(covar_cluster)
    cluster_weight = inv_covariance @ ra_cluster
    cluster_weight /= np.sum(cluster_weight)

    # Create a DataFrame to store results
    kmeans_df_cluster = pd.DataFrame({
        'Asset': assets_in_cluster,
        'kmeans_cluster_{}'.format(cluster_num): cluster_weight
    })
    cluster_dfs.append(kmeans_df_cluster)

# Concatenate DataFrames for all clusters
kmeans_df = pd.concat(cluster_dfs, ignore_index=True)



##Genetic Algorithm

import math
import random
import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga

# Calculate average returns for each asset
ra = np.mean(cdf_3, axis=0)

# Create a covariance matrix
covar = cdf_3.cov()

# Define the optimization function for Genetic Algorithm
def optimize_weights(weights):
    portfolio_return = np.sum(weights * ra)
    portfolio_volatility = math.sqrt(np.dot(weights.T, np.dot(covar, weights)))
    excess_return = portfolio_return - risk_free_rate
    sharpe_ratio = excess_return / portfolio_volatility
    return -sharpe_ratio  # Minimize negative of Sharpe ratio

# Load data and compute correlation matrix
corr_matrix = cdf_3.corr().values

# Set risk-free rate
risk_free_rate = cdf_3['ROI-Government SECs'].std()

# Set parameters for GA optimization
varbound = np.array([[0, 1]] * len(cdf_3.columns))

# Initialize Genetic Algorithm
model = ga(function=optimize_weights, dimension=len(cdf_3.columns), variable_type='real', variable_boundaries=varbound)

# Run Genetic Algorithm optimization
model.run()

# Retrieve optimized weights from the Genetic Algorithm
optimal_weights = model.output_dict['variable']

# Normalize the weights to sum up to 1
optimal_weights /= np.sum(optimal_weights)

# Compute portfolio Sharpe ratio
portfolio_return = np.sum(optimal_weights * ra)
portfolio_volatility = math.sqrt(np.dot(optimal_weights.T, np.dot(covar, optimal_weights)))
excess_return = portfolio_return - risk_free_rate
ga_portfolio_sharpe_ratio = excess_return / portfolio_volatility

# Calculate expected portfolio return using optimal weights and mean returns
expected_portfolio_return = np.sum(optimal_weights * ra)

# Print GA mean return for the portfolio
print("GA_Mean_Return:", expected_portfolio_return)

# Print the portfolio Sharpe ratio
print("Portfolio Sharpe Ratio (GA):", ga_portfolio_sharpe_ratio)

# Print list of assets and allocated weights
print("List of Assets and Allocated Weights (GA):")
for asset, weight in zip(cdf_3.columns, optimal_weights):
    print(f"{asset}: {weight:.2f}")

# Create a DataFrame to store results
ga_df_result = pd.DataFrame({
    'Asset': cdf_3.columns,
    'GA_Weight': optimal_weights
})



##Particle Swam Optimization

import math
import random
import numpy as np
import pandas as pd

class Particle:
    def __init__(self, num_assets):
        self.position = np.random.rand(num_assets)
        self.velocity = np.random.rand(num_assets)
        self.best_position = self.position.copy()
        self.best_fitness = float('-inf')

class ParticleSwarmOptimization:
    def __init__(self, returns, corr_matrix, risk_free, population_size, max_iterations):
        self.returns = returns
        self.corr_matrix = corr_matrix
        self.risk_free = risk_free
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.num_assets = returns.shape[1]
        self.population = [Particle(self.num_assets) for _ in range(population_size)]
        self.global_best_position = np.zeros(self.num_assets)
        self.global_best_fitness = float('-inf')

    def fitness(self, weights):
        mean_returns = np.mean(self.returns, axis=0)
        portfolio_return = np.sum(weights * mean_returns)
        covariance_matrix = np.cov(self.returns, rowvar=False)
        portfolio_volatility = math.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        excess_return = portfolio_return - self.risk_free
        sharpe_ratio = excess_return / portfolio_volatility
        return sharpe_ratio

    def update_particle_best(self, particle):
        particle_fitness = self.fitness(particle.position)
        if particle_fitness > particle.best_fitness:
            particle.best_position = particle.position.copy()
            particle.best_fitness = particle_fitness

    def update_global_best(self):
        for particle in self.population:
            if particle.best_fitness > self.global_best_fitness:
                self.global_best_position = particle.best_position.copy()
                self.global_best_fitness = particle.best_fitness

    def update_velocity(self, particle):
        inertia_weight = 0.5
        cognitive_weight = 1.5
        social_weight = 2.0
        r1 = random.random()
        r2 = random.random()
        cognitive_component = cognitive_weight * r1 * (particle.best_position - particle.position)
        social_component = social_weight * r2 * (self.global_best_position - particle.position)
        particle.velocity = inertia_weight * particle.velocity + cognitive_component + social_component

    def update_position(self, particle):
        particle.position += particle.velocity
        particle.position = np.clip(particle.position, 0, 1)

    def optimize(self):
        for _ in range(self.max_iterations):
            for particle in self.population:
                self.update_particle_best(particle)
            self.update_global_best()
            for particle in self.population:
                self.update_velocity(particle)
                self.update_position(particle)
        return self.global_best_position


# Load data and compute correlation matrix
corr_matrix = cdf_3.corr().values

# Set risk-free rate
risk_free_rate = cdf_3['ROI-Government SECs'].std()

# Set parameters for PSO optimization
population_size = 50
max_iterations = 10

# Create an instance of ParticleSwarmOptimization
pso = ParticleSwarmOptimization(cdf_3.values, corr_matrix, risk_free_rate, population_size, max_iterations)

# Run PSO optimization
optimal_weights = pso.optimize()

# Normalize the weights to sum up to 1
optimal_weights /= np.sum(optimal_weights)


# Compute portfolio Sharpe ratio
mean_returns = np.mean(cdf_3.values, axis=0)
portfolio_return = np.sum(optimal_weights * mean_returns)
covariance_matrix = np.cov(cdf_3.values, rowvar=False)
portfolio_volatility = math.sqrt(np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights)))
excess_return = portfolio_return - risk_free_rate
pso_portfolio_sharpe_ratio = excess_return / portfolio_volatility

# Calculate expected portfolio return using optimal weights and mean returns
expected_portfolio_return = np.sum(optimal_weights * np.mean(cdf_3.values, axis=0))

# Print PSO mean return for the portfolio
print("PSO_Mean_Return:", expected_portfolio_return)

# Print the portfolio Sharpe ratio
print("Portfolio Sharpe Ratio (PSO):", pso_portfolio_sharpe_ratio)

# Print list of assets and allocated weights
print("List of Assets and Allocated Weights (PSO):")
for asset, weight in zip(cdf_3.columns, optimal_weights):
    print(f"{asset}: {weight:.2f}")

# Create a DataFrame to store results
pso_df_result = pd.DataFrame({
    'Asset': cdf_3.columns,
    'PSO_Weight': optimal_weights
})


## SVM Classifier

import pickle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

# Load and preprocess data
# Assuming cdf_3 is already defined

# Set the originalRows, originalColumns
originalRows = 120   #excluding header
originalColumns = 9  #excluding date

# Extract asset labels
assetLabels = cdf_3.columns[1:originalColumns+1].tolist()

# Extract stock returns excluding header and trading dates
dfAssets_returns = cdf_3.iloc[0:, 1:]

# Store stock returns as an array
ardfAssets_returns = np.asarray(dfAssets_returns)

# Compute mean returns and variance-covariance matrix of returns
meanReturns = np.mean(ardfAssets_returns, axis=0)
covReturns = np.cov(ardfAssets_returns, rowvar=False)

# Prepare asset parameters for SVM classification
assetParameters = np.concatenate([meanReturns.reshape(-1, 1), covReturns], axis=1)

# Perform hyperparameter tuning for SVM using GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel': ['rbf', 'linear', 'poly']}
svm_grid = GridSearchCV(SVC(probability=True), param_grid, cv=2)
svm_grid.fit(assetParameters, labels)  # Assuming 'labels' from the previous KMeans clustering

# Get the best SVM model from hyperparameter tuning
best_svm = svm_grid.best_estimator_

# Predict labels for asset clusters using the best SVM model
svm_labels = best_svm.predict(assetParameters)

# Store asset labels with their respective cluster labels
cluster_assets = {}
for i, label in enumerate(svm_labels):
    if label not in cluster_assets:
        cluster_assets[label] = []
    cluster_assets[label].append(assetLabels[i])

# Iterate over clusters and allocate weights for each asset
for cluster_num, cluster_assets in cluster_assets.items():
    # Select data for the current cluster
    cluster_data = dfAssets_returns[cluster_assets]

    # Calculate mean returns, covariance matrix, and volatility for the current cluster
    ra_cluster = cluster_data.mean(axis=0)
    covar_cluster = cluster_data.cov()
    vols_cluster = np.sqrt(np.diagonal(covar_cluster))

    # Calculate weights using mean-variance optimization or any other method
    # Here, you can use any portfolio optimization technique like Mean-Variance Optimization, Risk Parity, etc.
    # For example, we can use the inverse covariance matrix approach
    inv_covariance = np.linalg.inv(covar_cluster)
    cluster_weight = inv_covariance @ ra_cluster
    cluster_weight /= np.sum(cluster_weight)

    # Create a DataFrame to store results
    svm_df_cluster = pd.DataFrame({
        'Asset': cluster_assets,
        'svm_Weight_cluster_{}'.format(cluster_num): cluster_weight
    })

    # Display the result DataFrame for each cluster
    print("Cluster", cluster_num)
    print(svm_df_cluster)


st.title("Asset Portfolio Recommendation App")

"""### All portfolios"""

cresult_df_mv = result_df_MV.set_index('Asset').T
ckmeans_df = kmeans_df.set_index('Asset').T
cpso_df_result = pso_df_result.set_index('Asset').T
csvm_df_cluster = svm_df_cluster.set_index('Asset').T
cga_df_result = ga_df_result.set_index('Asset').T

def combine_and_merge_weights(dfs, avg_return_col):
  """
  Combines multiple DataFrames with asset weights (rows) and an "Asset" column as the index, excludes a specified average return row,
  and merges them into a single DataFrame.

  Args:
      dfs (list): List of DataFrames containing asset weights as rows (indexed by "Asset").
      avg_return_col (str): Name of the row containing average return (to be excluded from merging).

  Returns:
      pd.DataFrame: The combined DataFrame with weights from each DataFrame (excluding average return).
  """

  # Get all unique asset labels from all DataFrames (as a list, not set)
  all_asset_labels = list(set.union(*[set(df.index) for df in dfs]))

  # Create the empty combined DataFrame with all asset labels as the index
  combined_df = pd.DataFrame(columns=dfs[0].columns, index=all_asset_labels)  # Use first DataFrame's columns

  # Iterate through each DataFrame and add weights to the combined DataFrame
  for df in dfs:
      # Extract and store the average return row (if present)
      avg_return = None
      if avg_return_col in df.index:
          avg_return = df.loc[avg_return_col]
          df = df.drop(avg_return_col)  # Remove avg_return row from weights

      # Fill missing assets with 0 (assuming weights might be missing)
      df = df.reindex(all_asset_labels, fill_value=0)  # Optional depending on your data

      # Add weights from the current DataFrame to the combined DataFrame
      combined_df = combined_df.add(df, fill_value=0)

  # Add the average return row back (if present) as the first row
  if avg_return is not None:
      combined_df.loc[avg_return_col] = avg_return

  return combined_df

# Assuming DataFrames are defined correctly and have "Asset" as the index
# Specify the column name containing average return (adjust if necessary)
avg_return_col = "Average Return"

# Combine the DataFrames (without normalization)
combined_weights_df = combine_and_merge_weights([cresult_df_mv, ckmeans_df, cpso_df_result, csvm_df_cluster, cga_df_result], avg_return_col)



# Set intersection of asset names (assuming unique asset names)
common_assets = set(combined_weights_df.columns) & set(ra.index)

# Check for missing assets (optional)
missing_assets = list(combined_weights_df.columns.difference(common_assets))
if missing_assets:
    print(f"Warning: Missing assets in combined_weights_df: {missing_assets}")

# Calculate portfolio mean for each row
combined_weights_df["portfolio_mean"] = np.nan  # Initialize with NaN for unmatched assets

for idx in combined_weights_df.index:
    matching_weights = combined_weights_df.loc[idx, list(common_assets)]  # Convert common_assets to list
    portfolio_mean = np.dot(matching_weights, ra.loc[list(common_assets)])  # Dot product for weighted sum
    combined_weights_df.loc[idx, "portfolio_mean"] = portfolio_mean

# Calculate portfolio Sharpe ratio for each row
combined_weights_df["portfolio_sharpe_ratios"] = np.nan  # Initialize with NaN

for idx in combined_weights_df.index:
    matching_weights = combined_weights_df.loc[idx, list(common_assets)]  # Convert common_assets to list
    portfolio_sharpe_ratio = np.dot(matching_weights, sharpe_ratios.loc[list(common_assets)])
    combined_weights_df.loc[idx, "portfolio_sharpe_ratios"] = portfolio_sharpe_ratio

# Print the updated DataFrame
st.dataframe(combined_weights_df)



import streamlit as st

# Dictionary to map original asset names to new names
asset_name_mapping = {
    'ROI-C & C Bonds': 'Commercial Corporate Bonds',
    'ROI-Demand DEPs': 'Demand Deposits',
    'ROI-GFs': 'Guaranteed Funds',
    'ROI-Government SECs': 'Government Securities',
    'ROI-IProperty': 'Immovable Property',
    'ROI-Offshore INVE': 'Offshore Investments',
    'ROI-P Unit Trusts': 'Property Unit Trusts',
    'ROI-Qouted EQ': 'Quoted Equity',
    'ROI-UNQouted EQ': 'Unquoted Equity'
}

## Function to recommend portfolios based on user input
def recommend_portfolios(expected_return, risk_level, asset_weights, combined_weights_df):
    # Filter portfolios based on user-specified expected return and risk level
    filtered_portfolios = combined_weights_df[
        (combined_weights_df['portfolio_mean'] >= expected_return)
    ]

    if risk_level == "Assertive":
        filtered_portfolios = filtered_portfolios[
            (filtered_portfolios['portfolio_sharpe_ratios'] >= filtered_portfolios['portfolio_sharpe_ratios'].quantile(0.75))
        ]
    elif risk_level == "Conservative":
        filtered_portfolios = filtered_portfolios[
            (filtered_portfolios['portfolio_sharpe_ratios'] <= filtered_portfolios['portfolio_sharpe_ratios'].quantile(0.25))
        ]
    else:  # Moderate risk
        average_sharpe_ratio = filtered_portfolios['portfolio_sharpe_ratios'].mean()
        filtered_portfolios = filtered_portfolios[
            (filtered_portfolios['portfolio_sharpe_ratios'] >= average_sharpe_ratio)
        ]

    # Filter portfolios based on user-specified weights for each asset
    for asset, weight in asset_weights.items():
        filtered_portfolios = filtered_portfolios[
            (filtered_portfolios[asset] <= weight)
        ]

    # Sort portfolios by Sharpe ratio in descending order
    recommended_portfolios = filtered_portfolios.sort_values(by='portfolio_sharpe_ratios', ascending=False).head(3)

    return recommended_portfolios

# Main function to run the Streamlit app
def main():
    # Set page title
    st.title("Recommended portfolio based on user requirements")

    # User input for expected portfolio return
    expected_return = st.slider("Expected Portfolio Return", min_value=0.0, max_value=0.2, step=0.01)

    # User input for risk level
    risk_level = st.selectbox("Risk Level", ["Assertive", "Conservative", "Moderate"])

    # Dictionary to store asset weights selected by the user
    asset_weights = {}
    for asset in ['Commercial Corporate Bonds', 'Demand Deposits', 'Guaranteed Funds', 'Government Securities', 'Immovable Property', 'Offshore Investments', 'Property Unit Trusts', 'Quoted Equity', 'Unquoted Equity']:
        original_asset_name = next(key for key, value in asset_name_mapping.items() if value == asset)
        weight = st.slider(f"Maximum Weight for {asset}", min_value=0.0, max_value=1.0, step=0.01, key=original_asset_name)
        asset_weights[original_asset_name] = weight

    # Button to trigger portfolio recommendation
    if st.button("Recommend Portfolios"):
        # Call function to recommend portfolios based on user input
        recommended_portfolios = recommend_portfolios(expected_return, risk_level, asset_weights, combined_weights_df)

        # Display recommended portfolios
        st.subheader("Recommended Portfolios:")
        st.write(recommended_portfolios)

# Run the Streamlit app
if __name__ == "__main__":
    main()