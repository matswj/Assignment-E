from load_data import load_data
from preprocess import preprocess_data
from train_model import train_logreg, train_mlp
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
data = pd.read_csv('C:/Users/matsw/OneDrive/Skrivebord/Intelligente Systemer/Assignment-E/Assignment-E/data/crop_recommendation.csv')


# Preprocess data
X_scaled, y = preprocess_data(data)

# Split data in training set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train and evaluate
print("Logistic Regression Results:")
print(train_logreg(X_train, y_train, X_test, y_test))

print("MLP Classifier Results:")
print(train_mlp(X_train, y_train, X_test, y_test))
