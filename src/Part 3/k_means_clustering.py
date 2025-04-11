import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('C:/Users/matsw/OneDrive/Skrivebord/Intelligente Systemer/Assignment-E/Assignment-E/data/crop_recommendation.csv')

# Select relevant features (drop 'label' since we're doing unsupervised learning)
X = data.drop(columns=['label'])

# Standardize the features for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# List of k values to experiment with
k_values = [2, 3, 4, 5, 6]

# Create a figure to plot the results
plt.figure(figsize=(10, 6))

# For each k value, fit K-Means and calculate Silhouette Score
silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    
    # Calculate silhouette score
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)
    
    # Plot the clusters
    plt.subplot(2, 3, k-1)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.title(f'K-Means with k={k}')

# Plot the silhouette scores for each k value
plt.figure(figsize=(8, 6))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', color='b')
plt.title('Silhouette Scores for Different k Values')
plt.xlabel('k (Number of Clusters)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()
