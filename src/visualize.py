import matplotlib.pyplot as plt

def plot_anomaly_distribution(anomaly_types_distribution):
    labels, counts = zip(*anomaly_types_distribution.items())
    plt.bar(labels, counts)
    plt.xlabel('Anomaly Types')
    plt.ylabel('Counts')
    plt.title('Anomaly Type Distribution')
    plt.show()
