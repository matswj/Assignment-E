import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:/Users/matsw/OneDrive/Skrivebord/Intelligente Systemer/Assignment-E/Assignment-E/data/crop_recommendation.csv')


# Plot histograms for each feature
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
for feature in features:
    plt.figure(figsize=(8, 6))
    plt.hist(data[feature], bins=20, edgecolor='black', alpha=0.7)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Boxplots to check for outliers
for feature in features:
    plt.figure(figsize=(8, 6))
    plt.boxplot(data[feature], vert=False)
    plt.title(f"Boxplot of {feature}")
    plt.xlabel(feature)
    plt.show()

# Correlation heatmap to visualize relationships among features
correlation_matrix = data[features].corr()
plt.figure(figsize=(10, 8))
plt.matshow(correlation_matrix, fignum=1, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(features)), features, rotation=90)
plt.yticks(range(len(features)), features)
plt.title("Correlation Heatmap of Features")
plt.show()
