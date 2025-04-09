from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
data = pd.read_csv('C:/Users/matsw/OneDrive/Skrivebord/Intelligente Systemer/Assignment-E/Assignment-E/data/crop_recommendation.csv')
X = data.drop('label', axis=1) 
y = data['label']  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning (testing different values for C)
for C_value in [0.01, 1, 100]:
    logreg = LogisticRegression(C=C_value, max_iter=1000)
    logreg.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = logreg.predict(X_test)
    print(f"Results for C = {C_value}:")
    print(classification_report(y_test, y_pred))
