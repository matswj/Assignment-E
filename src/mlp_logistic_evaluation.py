from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
data = pd.read_csv('C:/Users/matsw/OneDrive/Skrivebord/Intelligente Systemer/Assignment-E/Assignment-E/data/crop_recommendation.csv')
X = data.drop('label', axis=1)
y = data['label']  # Target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Hyperparameter Tuning (testing different values of C, solvers, penalties, and max_iter)
print("\nLogistic Regression Results:")
for C_value in [0.01, 1, 100]:
    for solver in ['liblinear', 'saga', 'newton-cg']:
        # If using 'newton-cg', only use 'l2' penalty
        if solver == 'newton-cg':
            penalty_values = ['l2']
        else:
            penalty_values = ['l1', 'l2']

        for penalty in penalty_values:
            logreg = LogisticRegression(C=C_value, max_iter=1000, solver=solver, penalty=penalty)
            logreg.fit(X_train, y_train)

            # Evaluate the model
            y_pred = logreg.predict(X_test)
            print(f"Results for C = {C_value}, Solver = {solver}, Penalty = {penalty}:")
            print(classification_report(y_test, y_pred))

# MLP Classifier Hyperparameter Tuning (testing different hidden_layer_sizes, activation functions, and learning rates)
print("\nMLP Classifier Results:")
for hidden_size in [(50,), (100,), (100, 50)]:
    for activation_func in ['relu', 'tanh', 'logistic']:
        for learning_rate in [0.001, 0.01, 0.1]:  # Different learning rates
            for solver in ['adam', 'sgd', 'lbfgs']:  # Experiment with different solvers
                mlp = MLPClassifier(hidden_layer_sizes=hidden_size, activation=activation_func, 
                                    max_iter=1000, learning_rate_init=learning_rate, solver=solver)
                mlp.fit(X_train, y_train)

                # Evaluate the model
                y_pred = mlp.predict(X_test)
                print(f"Results for hidden_layer_sizes = {hidden_size}, activation = {activation_func}, "
                      f"learning_rate = {learning_rate}, solver = {solver}:")
                print(classification_report(y_test, y_pred))
