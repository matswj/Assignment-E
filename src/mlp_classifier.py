from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# Load data
data = pd.read_csv('C:/Users/matsw/OneDrive/Skrivebord/Intelligente Systemer/Assignment-E/Assignment-E/data/crop_recommendation.csv')
X = data.drop('label', axis=1) 
y = data['label']  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for MLP
for hidden_size in [(50,), (100,), (100, 50)]:
    for activation_func in ['relu', 'tanh', 'logistic']:
        mlp = MLPClassifier(hidden_layer_sizes=hidden_size, activation=activation_func, max_iter=1000)
        mlp.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = mlp.predict(X_test)
        print(f"Results for hidden_layer_sizes = {hidden_size}, activation = {activation_func}:")
        print(classification_report(y_test, y_pred))
