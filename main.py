import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import functions as fn
import classifiers as cl
import data_visualization as dv
from sklearn.cluster import AgglomerativeClustering, KMeans

## Dataset parameters
SUBJECTS = 22
TRAIN_SUBJECTS = None
TEST_SUBJECTS = None

## RF parameters
N_TREES = 100

## Neural network parameters
EPOCHS = 20
BATCH_SIZE = 10


def main():
    # X, y = fn.read_and_preprocess_data(subjects=SUBJECTS)
    X, y = fn.load_preprocessed_data()

    ####### Add data visualization here (?)

    ####### Train classifiers and evaluate them using Leave-One-Out Cross-Validation (Very computationally expensive!)

    # # Evaluate Neural Network classifier
    # cl.custom_loo_cv("Sequential", X, y, EPOCHS, BATCH_SIZE)

    # # Evaluate Random Forest classifier
    # cl.custom_loo_cv("RandomForest", X, y, n_trees=N_TREES)

    # # Evaluate Bayesian Networks classifier
    # cl.custom_loo_cv("BayesianNetworks", X, y)

    ####### Train classifiers and evaluate them WITHOUT using Leave-One-Out Cross-Validation
    cl.train_and_evaluate_models_no_cv(X, y, EPOCHS, BATCH_SIZE, N_TREES)

    ####### Clustering
    # Create feature matrix
    feature_matrix = fn.create_feature_matrix(y)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=17)
    km_labels = kmeans.fit_predict(feature_matrix)
    dv.plot_clusters(feature_matrix, km_labels)
    dv.plot_label_distr_per_cluster(y, km_labels)

    # Perform hierarchical clustering
    agg_clustering = AgglomerativeClustering(n_clusters=3)
    agg_labels = agg_clustering.fit_predict(feature_matrix)
    dv.plot_clusters(feature_matrix, agg_labels)
    dv.plot_dendogram(feature_matrix, agg_labels)
    dv.plot_label_distr_per_cluster(y, agg_labels)


if __name__ == "__main__":
    main()
