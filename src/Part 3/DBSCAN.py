from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('C:/Users/matsw/OneDrive/Skrivebord/Intelligente Systemer/Assignment-E/Assignment-E/data/crop_recommendation.csv')

# Select relevant features (drop 'label' since we're doing unsupervised learning)
X = data.drop(columns=['label'])

# Standardize the features for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Experiment with different eps and min_samples
eps_values = [0.3, 0.5, 0.7]
min_samples_values = [3, 5, 7]

for eps in eps_values:
    for min_samples in min_samples_values:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        
        # Plot results or analyze clusters
        plt.scatter(X['N'], X['P'], c=labels, cmap='viridis')
        plt.title(f'DBSCAN with eps={eps} and min_samples={min_samples}')
        plt.xlabel('N (Nitrogen)')
        plt.ylabel('P (Phosphorus)')
        plt.show()
        
        print(f"DBSCAN labels for eps={eps}, min_samples={min_samples}: {labels}")