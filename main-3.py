import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

# Load the data into a pandas dataframe
data = pd.read_csv('CO2emissions(kg_perPPP$_of_GDP.csv')

# Select the columns we want to cluster
X = data.iloc[:, 1:]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster the data using K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Add the cluster labels to the dataframe
data['cluster'] = kmeans.labels_

# Plot the clusters
plt.scatter(data['European Union'], data['United States'], c=data['cluster'])
plt.xlabel('European Union')
plt.ylabel('United States')

# Plot the cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 5], marker='*', s=150, linewidths=3, color='r')

# Save the plot as a PNG file
plt.savefig('CO2Emissions_Clustering.png')

# Extract US data
us_data = data['United States']

# Create array of years
years = np.arange(2002, 2020)

# Fit polynomial regression to US data
coefficients = np.polyfit(years, us_data, 3)
poly = np.poly1d(coefficients)

# Predict CO2 emissions for next 10 years
future_years = np.arange(2020, 2030)
predicted_data = poly(future_years)

# Plot data, fitted function, and confidence range
fig, ax = plt.subplots()
ax.plot(years, us_data, '-', label='Data', color='red')
ax.plot(future_years, predicted_data, label='Prediction')
ax.legend()
ax.set_xlabel('Year')
ax.set_ylabel('CO2 Emissions (kg per PPP $ of GDP)')

# Save the plot as a PNG file
plt.savefig('CO2Emissions_Prediction.png')
