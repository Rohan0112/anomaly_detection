from src.predict import process_large_file
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly Detection and Classification")
    parser.add_argument("--file", required=True, help="Path to the CSV file containing CAN data")
    parser.add_argument("--threshold", type=float, default=0.05, help="Reconstruction loss threshold for anomalies")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Chunk size for processing large datasets")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of KMeans clusters")
    args = parser.parse_args()

    # Process the input file
    anomaly_count, anomaly_types_distribution = process_large_file(
        args.file, 
        chunk_size=args.chunk_size, 
        threshold=args.threshold, 
        n_clusters=args.n_clusters
    )

    print(f"Total Anomalies Detected: {anomaly_count}")
    print("Anomaly Types Distribution:")
    for anomaly_type, count in anomaly_types_distribution.items():
        print(f"  Type {anomaly_type}: {count}")
