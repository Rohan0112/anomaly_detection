import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
from src.preprocess import preprocess_chunk
import pickle
from sklearn.exceptions import NotFittedError

# Load the pre-trained autoencoder and scaler
autoencoder = load_model('models/autoencoder_model.keras')
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

def detect_anomalies_and_classify(chunk_normalized, threshold, autoencoder, kmeans):
    # Get the reconstruction error for each data point
    reconstruction = autoencoder.predict(chunk_normalized)
    reconstruction_error = np.mean(np.abs(reconstruction - chunk_normalized), axis=1)
    
    # Detect anomalies based on reconstruction error
    anomalies = reconstruction_error > threshold  # Compare error with threshold
    anomaly_scores = reconstruction_error  # Or use another measure to represent anomaly severity
    
    # If anomalies detected, classify them using KMeans
    if np.any(anomalies):
        # Clustering anomalies to classify them (e.g., DoS, fuzzy, etc.)
        anomaly_data = chunk_normalized[anomalies]  # Get data that is anomalous
        anomaly_types = kmeans.predict(anomaly_data)  # Classify anomalies using KMeans
    else:
        anomaly_types = []
    
    return anomalies, anomaly_scores, anomaly_types

def process_large_file(file_path):
    # Load the dataset (assuming CSV, adapt if necessary)
    df = pd.read_csv(file_path)

    # Assuming 'data_columns' are the features in your dataset
    data_columns = ['Data_0', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7']  # Replace with your actual columns
    # Ensure 'ID' column is properly converted from hex (if necessary)
    df['ID'] = df['ID'].apply(lambda x: int(x, 16) if isinstance(x, str) else x)

# Now apply the scaling to the numerical columns  # Fit scaler on the dataset

    # Normalize data
    scaler = StandardScaler()
    scaler.fit(df[data_columns])  # Fit scaler on the dataset
    
    # Load the trained autoencoder and KMeans model
    autoencoder = load_model('models/autoencoder_model.keras')  # Replace with your model path
    kmeans = KMeans(n_clusters=3)  # Replace with the number of clusters for classification
    kmeans.fit(df[data_columns])
    # Process chunks of data
    chunk_size = 1000  # Define the chunk size
    anomaly_count = 0
    type_distribution = {0: 0, 1: 0, 2: 0}  # To store distribution of anomalies (e.g., DoS, fuzzy)
    
    for i in range(0, len(df), chunk_size):
        chunk = df[i:i+chunk_size]  # Get a chunk of the data
        chunk_normalized = preprocess_chunk(chunk, scaler)  # Normalize the chunk

        # Set a threshold for detecting anomalies (you may want to experiment with this)
        threshold = 0.1  # Example threshold, adjust based on your dataset
        
        # Detect anomalies and classify
        anomalies, anomaly_scores, anomaly_types = detect_anomalies_and_classify(chunk_normalized, threshold, autoencoder, kmeans)
        
        # Count anomalies
        anomaly_count += np.sum(anomalies)  # Count anomalies in this chunk
        for t in anomaly_types:
            type_distribution[t] += 1  # Update the type distribution (DoS, fuzzy, etc.)

    return anomaly_count, type_distribution