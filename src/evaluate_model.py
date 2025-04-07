import pandas as pd
from sklearn.model_selection import train_test_split
from train_model import train_logreg, train_mlp  # Importing the functions

# Load data
data = pd.read_csv('data/crop_recommendation.csv')
X = data.drop('label', axis=1)  # Features
y = data['label']  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate Logistic Regression
print("Logistic Regression Results:")
logreg_report = train_logreg(X_train, y_train, X_test, y_test)
print(logreg_report)

# Train and evaluate MLP Classifier
print("MLP Classifier Results:")
mlp_report = train_mlp(X_train, y_train, X_test, y_test)
print(mlp_report)
