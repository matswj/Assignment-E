from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(data):
    # Label Encoding for 'label' column
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])

    # Split into features (X) and target (y)
    X = data.drop('label', axis=1)
    y = data['label']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y