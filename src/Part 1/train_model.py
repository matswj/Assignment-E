from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

def train_logreg(X_train, y_train, X_test, y_test):
    """
    Trains a Logistic Regression model and evaluates it using classification report.

    Parameters:
        X_train (array-like): Training features
        y_train (array-like): Training labels
        X_test (array-like): Test features
        y_test (array-like): Test labels

    Returns:
        str: Classification report
    """
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    
    # Predict on test set
    y_pred_logreg = logreg.predict(X_test)
    
    return classification_report(y_test, y_pred_logreg)

def train_mlp(X_train, y_train, X_test, y_test):
    """
    Trains an MLP Classifier model and evaluates it using classification report.

    Parameters:
        X_train (array-like): Training features
        y_train (array-like): Training labels
        X_test (array-like): Test features
        y_test (array-like): Test labels

    Returns:
        str: Classification report
    """
    mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    
    # Predict on test set
    y_pred_mlp = mlp.predict(X_test)
    
    return classification_report(y_test, y_pred_mlp)