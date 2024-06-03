import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from functions import load_preprocessed_data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

def create_feature_matrix(y: np.array):
    num_subjects = len(y)
    num_labels = 12 
    # Initialize the feature matrix
    feature_matrix = np.zeros((num_subjects, num_labels))
    # Fill the feature matrix
    for i, subject_data in enumerate(y):
        labels, counts = np.unique(subject_data, return_counts=True)
        feature_matrix[i, labels] = counts

    
    # Normalize the feature vectors
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    return feature_matrix

def plot_clusters(feature_matrix, labels):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(feature_matrix)

    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Clustering of Individuals Based on Activities')
    plt.show()

def plot_dendogram(feature_matrix, labels):
    # Create a linkage matrix for dendrogram
    linkage_matrix = linkage(feature_matrix, method='ward')

    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, labels=np.arange(feature_matrix.shape[0]))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()

    plot_clusters(feature_matrix, labels)