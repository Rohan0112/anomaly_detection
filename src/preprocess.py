import pandas as pd
import numpy as np

def preprocess_chunk(chunk, scaler):
    # Identify columns containing data
    data_columns = [col for col in chunk.columns if 'Data_' in col]
    data_columns = sorted(data_columns)  # Ensure consistent column order

    # Handle missing values and convert to numeric
    chunk[data_columns] = chunk[data_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Check if data is valid
    if chunk[data_columns].empty:
        print("Warning: Empty or invalid data detected in chunk.")
        return np.empty((0, len(data_columns)))  # Return an empty array with correct shape

    # Normalize using the pre-trained scaler
    chunk_normalized = scaler.transform(chunk[data_columns])
    return chunk_normalized

