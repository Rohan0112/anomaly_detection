from sklearn.ensemble import IsolationForest
import joblib

def train_model(data):
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(data)
    joblib.dump(model, '../models/autoencoder_model.keras')
    print("Model trained and saved.")
