from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

def train_logreg(X_train, y_train, X_test, y_test):
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred_logreg = logreg.predict(X_test)
    return classification_report(y_test, y_pred_logreg)

def train_mlp(X_train, y_train, X_test, y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    return classification_report(y_test, y_pred_mlp)
